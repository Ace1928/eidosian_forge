from __future__ import absolute_import
import re
import sys
import copy
import codecs
import itertools
from . import TypeSlots
from .ExprNodes import not_a_constant
import cython
from . import Nodes
from . import ExprNodes
from . import PyrexTypes
from . import Visitor
from . import Builtin
from . import UtilNodes
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .StringEncoding import EncodedString, bytes_literal, encoded_string
from .Errors import error, warning
from .ParseTreeTransforms import SkipDeclarations
from .. import Utils
class EarlyReplaceBuiltinCalls(Visitor.EnvTransform):
    """Optimize some common calls to builtin types *before* the type
    analysis phase and *after* the declarations analysis phase.

    This transform cannot make use of any argument types, but it can
    restructure the tree in a way that the type analysis phase can
    respond to.

    Introducing C function calls here may not be a good idea.  Move
    them to the OptimizeBuiltinCalls transform instead, which runs
    after type analysis.
    """
    visit_Node = Visitor.VisitorTransform.recurse_to_children

    def visit_SimpleCallNode(self, node):
        self.visitchildren(node)
        function = node.function
        if not self._function_is_builtin_name(function):
            return node
        return self._dispatch_to_handler(node, function, node.args)

    def visit_GeneralCallNode(self, node):
        self.visitchildren(node)
        function = node.function
        if not self._function_is_builtin_name(function):
            return node
        arg_tuple = node.positional_args
        if not isinstance(arg_tuple, ExprNodes.TupleNode):
            return node
        args = arg_tuple.args
        return self._dispatch_to_handler(node, function, args, node.keyword_args)

    def _function_is_builtin_name(self, function):
        if not function.is_name:
            return False
        env = self.current_env()
        entry = env.lookup(function.name)
        if entry is not env.builtin_scope().lookup_here(function.name):
            return False
        return True

    def _dispatch_to_handler(self, node, function, args, kwargs=None):
        if kwargs is None:
            handler_name = '_handle_simple_function_%s' % function.name
        else:
            handler_name = '_handle_general_function_%s' % function.name
        handle_call = getattr(self, handler_name, None)
        if handle_call is not None:
            if kwargs is None:
                return handle_call(node, args)
            else:
                return handle_call(node, args, kwargs)
        return node

    def _inject_capi_function(self, node, cname, func_type, utility_code=None):
        node.function = ExprNodes.PythonCapiFunctionNode(node.function.pos, node.function.name, cname, func_type, utility_code=utility_code)

    def _error_wrong_arg_count(self, function_name, node, args, expected=None):
        if not expected:
            arg_str = ''
        elif isinstance(expected, basestring) or expected > 1:
            arg_str = '...'
        elif expected == 1:
            arg_str = 'x'
        else:
            arg_str = ''
        if expected is not None:
            expected_str = 'expected %s, ' % expected
        else:
            expected_str = ''
        error(node.pos, '%s(%s) called with wrong number of args, %sfound %d' % (function_name, arg_str, expected_str, len(args)))

    def _handle_simple_function_float(self, node, pos_args):
        if not pos_args:
            return ExprNodes.FloatNode(node.pos, value='0.0')
        if len(pos_args) > 1:
            self._error_wrong_arg_count('float', node, pos_args, 1)
        arg_type = getattr(pos_args[0], 'type', None)
        if arg_type in (PyrexTypes.c_double_type, Builtin.float_type):
            return pos_args[0]
        return node

    def _handle_simple_function_slice(self, node, pos_args):
        arg_count = len(pos_args)
        start = step = None
        if arg_count == 1:
            stop, = pos_args
        elif arg_count == 2:
            start, stop = pos_args
        elif arg_count == 3:
            start, stop, step = pos_args
        else:
            self._error_wrong_arg_count('slice', node, pos_args)
            return node
        return ExprNodes.SliceNode(node.pos, start=start or ExprNodes.NoneNode(node.pos), stop=stop, step=step or ExprNodes.NoneNode(node.pos))

    def _handle_simple_function_ord(self, node, pos_args):
        """Unpack ord('X').
        """
        if len(pos_args) != 1:
            return node
        arg = pos_args[0]
        if isinstance(arg, (ExprNodes.UnicodeNode, ExprNodes.BytesNode)):
            if len(arg.value) == 1:
                return ExprNodes.IntNode(arg.pos, type=PyrexTypes.c_long_type, value=str(ord(arg.value)), constant_result=ord(arg.value))
        elif isinstance(arg, ExprNodes.StringNode):
            if arg.unicode_value and len(arg.unicode_value) == 1 and (ord(arg.unicode_value) <= 255):
                return ExprNodes.IntNode(arg.pos, type=PyrexTypes.c_int_type, value=str(ord(arg.unicode_value)), constant_result=ord(arg.unicode_value))
        return node

    def _handle_simple_function_all(self, node, pos_args):
        """Transform

        _result = all(p(x) for L in LL for x in L)

        into

        for L in LL:
            for x in L:
                if not p(x):
                    return False
        else:
            return True
        """
        return self._transform_any_all(node, pos_args, False)

    def _handle_simple_function_any(self, node, pos_args):
        """Transform

        _result = any(p(x) for L in LL for x in L)

        into

        for L in LL:
            for x in L:
                if p(x):
                    return True
        else:
            return False
        """
        return self._transform_any_all(node, pos_args, True)

    def _transform_any_all(self, node, pos_args, is_any):
        if len(pos_args) != 1:
            return node
        if not isinstance(pos_args[0], ExprNodes.GeneratorExpressionNode):
            return node
        gen_expr_node = pos_args[0]
        generator_body = gen_expr_node.def_node.gbody
        loop_node = generator_body.body
        yield_expression, yield_stat_node = _find_single_yield_expression(loop_node)
        if yield_expression is None:
            return node
        if is_any:
            condition = yield_expression
        else:
            condition = ExprNodes.NotNode(yield_expression.pos, operand=yield_expression)
        test_node = Nodes.IfStatNode(yield_expression.pos, else_clause=None, if_clauses=[Nodes.IfClauseNode(yield_expression.pos, condition=condition, body=Nodes.ReturnStatNode(node.pos, value=ExprNodes.BoolNode(yield_expression.pos, value=is_any, constant_result=is_any)))])
        loop_node.else_clause = Nodes.ReturnStatNode(node.pos, value=ExprNodes.BoolNode(yield_expression.pos, value=not is_any, constant_result=not is_any))
        Visitor.recursively_replace_node(gen_expr_node, yield_stat_node, test_node)
        return ExprNodes.InlinedGeneratorExpressionNode(gen_expr_node.pos, gen=gen_expr_node, orig_func='any' if is_any else 'all')
    PySequence_List_func_type = PyrexTypes.CFuncType(Builtin.list_type, [PyrexTypes.CFuncTypeArg('it', PyrexTypes.py_object_type, None)])

    def _handle_simple_function_sorted(self, node, pos_args):
        """Transform sorted(genexpr) and sorted([listcomp]) into
        [listcomp].sort().  CPython just reads the iterable into a
        list and calls .sort() on it.  Expanding the iterable in a
        listcomp is still faster and the result can be sorted in
        place.
        """
        if len(pos_args) != 1:
            return node
        arg = pos_args[0]
        if isinstance(arg, ExprNodes.ComprehensionNode) and arg.type is Builtin.list_type:
            list_node = arg
            loop_node = list_node.loop
        elif isinstance(arg, ExprNodes.GeneratorExpressionNode):
            gen_expr_node = arg
            loop_node = gen_expr_node.loop
            yield_statements = _find_yield_statements(loop_node)
            if not yield_statements:
                return node
            list_node = ExprNodes.InlinedGeneratorExpressionNode(node.pos, gen_expr_node, orig_func='sorted', comprehension_type=Builtin.list_type)
            for yield_expression, yield_stat_node in yield_statements:
                append_node = ExprNodes.ComprehensionAppendNode(yield_expression.pos, expr=yield_expression, target=list_node.target)
                Visitor.recursively_replace_node(gen_expr_node, yield_stat_node, append_node)
        elif arg.is_sequence_constructor:
            list_node = loop_node = arg.as_list()
        else:
            list_node = loop_node = ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_PySequence_ListKeepNew' if arg.is_temp and arg.type in (PyrexTypes.py_object_type, Builtin.list_type) else 'PySequence_List', self.PySequence_List_func_type, args=pos_args, is_temp=True)
        result_node = UtilNodes.ResultRefNode(pos=loop_node.pos, type=Builtin.list_type, may_hold_none=False)
        list_assign_node = Nodes.SingleAssignmentNode(node.pos, lhs=result_node, rhs=list_node, first=True)
        sort_method = ExprNodes.AttributeNode(node.pos, obj=result_node, attribute=EncodedString('sort'), needs_none_check=False)
        sort_node = Nodes.ExprStatNode(node.pos, expr=ExprNodes.SimpleCallNode(node.pos, function=sort_method, args=[]))
        sort_node.analyse_declarations(self.current_env())
        return UtilNodes.TempResultFromStatNode(result_node, Nodes.StatListNode(node.pos, stats=[list_assign_node, sort_node]))

    def __handle_simple_function_sum(self, node, pos_args):
        """Transform sum(genexpr) into an equivalent inlined aggregation loop.
        """
        if len(pos_args) not in (1, 2):
            return node
        if not isinstance(pos_args[0], (ExprNodes.GeneratorExpressionNode, ExprNodes.ComprehensionNode)):
            return node
        gen_expr_node = pos_args[0]
        loop_node = gen_expr_node.loop
        if isinstance(gen_expr_node, ExprNodes.GeneratorExpressionNode):
            yield_expression, yield_stat_node = _find_single_yield_expression(loop_node)
            yield_expression = None
            if yield_expression is None:
                return node
        else:
            yield_stat_node = gen_expr_node.append
            yield_expression = yield_stat_node.expr
            try:
                if not yield_expression.is_literal or not yield_expression.type.is_int:
                    return node
            except AttributeError:
                return node
        if len(pos_args) == 1:
            start = ExprNodes.IntNode(node.pos, value='0', constant_result=0)
        else:
            start = pos_args[1]
        result_ref = UtilNodes.ResultRefNode(pos=node.pos, type=PyrexTypes.py_object_type)
        add_node = Nodes.SingleAssignmentNode(yield_expression.pos, lhs=result_ref, rhs=ExprNodes.binop_node(node.pos, '+', result_ref, yield_expression))
        Visitor.recursively_replace_node(gen_expr_node, yield_stat_node, add_node)
        exec_code = Nodes.StatListNode(node.pos, stats=[Nodes.SingleAssignmentNode(start.pos, lhs=UtilNodes.ResultRefNode(pos=node.pos, expression=result_ref), rhs=start, first=True), loop_node])
        return ExprNodes.InlinedGeneratorExpressionNode(gen_expr_node.pos, loop=exec_code, result_node=result_ref, expr_scope=gen_expr_node.expr_scope, orig_func='sum', has_local_scope=gen_expr_node.has_local_scope)

    def _handle_simple_function_min(self, node, pos_args):
        return self._optimise_min_max(node, pos_args, '<')

    def _handle_simple_function_max(self, node, pos_args):
        return self._optimise_min_max(node, pos_args, '>')

    def _optimise_min_max(self, node, args, operator):
        """Replace min(a,b,...) and max(a,b,...) by explicit comparison code.
        """
        if len(args) <= 1:
            if len(args) == 1 and args[0].is_sequence_constructor:
                args = args[0].args
            if len(args) <= 1:
                return node
        cascaded_nodes = list(map(UtilNodes.ResultRefNode, args[1:]))
        last_result = args[0]
        for arg_node in cascaded_nodes:
            result_ref = UtilNodes.ResultRefNode(last_result)
            last_result = ExprNodes.CondExprNode(arg_node.pos, true_val=arg_node, false_val=result_ref, test=ExprNodes.PrimaryCmpNode(arg_node.pos, operand1=arg_node, operator=operator, operand2=result_ref))
            last_result = UtilNodes.EvalWithTempExprNode(result_ref, last_result)
        for ref_node in cascaded_nodes[::-1]:
            last_result = UtilNodes.EvalWithTempExprNode(ref_node, last_result)
        return last_result

    def _DISABLED_handle_simple_function_tuple(self, node, pos_args):
        if not pos_args:
            return ExprNodes.TupleNode(node.pos, args=[], constant_result=())
        result = self._transform_list_set_genexpr(node, pos_args, Builtin.list_type)
        if result is not node:
            return ExprNodes.AsTupleNode(node.pos, arg=result)
        return node

    def _handle_simple_function_frozenset(self, node, pos_args):
        """Replace frozenset([...]) by frozenset((...)) as tuples are more efficient.
        """
        if len(pos_args) != 1:
            return node
        if pos_args[0].is_sequence_constructor and (not pos_args[0].args):
            del pos_args[0]
        elif isinstance(pos_args[0], ExprNodes.ListNode):
            pos_args[0] = pos_args[0].as_tuple()
        return node

    def _handle_simple_function_list(self, node, pos_args):
        if not pos_args:
            return ExprNodes.ListNode(node.pos, args=[], constant_result=[])
        return self._transform_list_set_genexpr(node, pos_args, Builtin.list_type)

    def _handle_simple_function_set(self, node, pos_args):
        if not pos_args:
            return ExprNodes.SetNode(node.pos, args=[], constant_result=set())
        return self._transform_list_set_genexpr(node, pos_args, Builtin.set_type)

    def _transform_list_set_genexpr(self, node, pos_args, target_type):
        """Replace set(genexpr) and list(genexpr) by an inlined comprehension.
        """
        if len(pos_args) > 1:
            return node
        if not isinstance(pos_args[0], ExprNodes.GeneratorExpressionNode):
            return node
        gen_expr_node = pos_args[0]
        loop_node = gen_expr_node.loop
        yield_statements = _find_yield_statements(loop_node)
        if not yield_statements:
            return node
        result_node = ExprNodes.InlinedGeneratorExpressionNode(node.pos, gen_expr_node, orig_func='set' if target_type is Builtin.set_type else 'list', comprehension_type=target_type)
        for yield_expression, yield_stat_node in yield_statements:
            append_node = ExprNodes.ComprehensionAppendNode(yield_expression.pos, expr=yield_expression, target=result_node.target)
            Visitor.recursively_replace_node(gen_expr_node, yield_stat_node, append_node)
        return result_node

    def _handle_simple_function_dict(self, node, pos_args):
        """Replace dict( (a,b) for ... ) by an inlined { a:b for ... }
        """
        if len(pos_args) == 0:
            return ExprNodes.DictNode(node.pos, key_value_pairs=[], constant_result={})
        if len(pos_args) > 1:
            return node
        if not isinstance(pos_args[0], ExprNodes.GeneratorExpressionNode):
            return node
        gen_expr_node = pos_args[0]
        loop_node = gen_expr_node.loop
        yield_statements = _find_yield_statements(loop_node)
        if not yield_statements:
            return node
        for yield_expression, _ in yield_statements:
            if not isinstance(yield_expression, ExprNodes.TupleNode):
                return node
            if len(yield_expression.args) != 2:
                return node
        result_node = ExprNodes.InlinedGeneratorExpressionNode(node.pos, gen_expr_node, orig_func='dict', comprehension_type=Builtin.dict_type)
        for yield_expression, yield_stat_node in yield_statements:
            append_node = ExprNodes.DictComprehensionAppendNode(yield_expression.pos, key_expr=yield_expression.args[0], value_expr=yield_expression.args[1], target=result_node.target)
            Visitor.recursively_replace_node(gen_expr_node, yield_stat_node, append_node)
        return result_node

    def _handle_general_function_dict(self, node, pos_args, kwargs):
        """Replace dict(a=b,c=d,...) by the underlying keyword dict
        construction which is done anyway.
        """
        if len(pos_args) > 0:
            return node
        if not isinstance(kwargs, ExprNodes.DictNode):
            return node
        return kwargs