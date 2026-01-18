from __future__ import absolute_import
import cython
import copy
import hashlib
import sys
from . import PyrexTypes
from . import Naming
from . import ExprNodes
from . import Nodes
from . import Options
from . import Builtin
from . import Errors
from .Visitor import VisitorTransform, TreeVisitor
from .Visitor import CythonTransform, EnvTransform, ScopeTrackingTransform
from .UtilNodes import LetNode, LetRefNode
from .TreeFragment import TreeFragment
from .StringEncoding import EncodedString, _unicode
from .Errors import error, warning, CompileError, InternalError
from .Code import UtilityCode
class PostParse(ScopeTrackingTransform):
    """
    Basic interpretation of the parse tree, as well as validity
    checking that can be done on a very basic level on the parse
    tree (while still not being a problem with the basic syntax,
    as such).

    Specifically:
    - Default values to cdef assignments are turned into single
    assignments following the declaration (everywhere but in class
    bodies, where they raise a compile error)

    - Interpret some node structures into Python runtime values.
    Some nodes take compile-time arguments (currently:
    TemplatedTypeNode[args] and __cythonbufferdefaults__ = {args}),
    which should be interpreted. This happens in a general way
    and other steps should be taken to ensure validity.

    Type arguments cannot be interpreted in this way.

    - For __cythonbufferdefaults__ the arguments are checked for
    validity.

    TemplatedTypeNode has its directives interpreted:
    Any first positional argument goes into the "dtype" attribute,
    any "ndim" keyword argument goes into the "ndim" attribute and
    so on. Also it is checked that the directive combination is valid.
    - __cythonbufferdefaults__ attributes are parsed and put into the
    type information.

    Note: Currently Parsing.py does a lot of interpretation and
    reorganization that can be refactored into this transform
    if a more pure Abstract Syntax Tree is wanted.

    - Some invalid uses of := assignment expressions are detected
    """

    def __init__(self, context):
        super(PostParse, self).__init__(context)
        self.specialattribute_handlers = {'__cythonbufferdefaults__': self.handle_bufferdefaults}

    def visit_LambdaNode(self, node):
        collector = YieldNodeCollector()
        collector.visitchildren(node.result_expr)
        if collector.has_yield or collector.has_await or isinstance(node.result_expr, ExprNodes.YieldExprNode):
            body = Nodes.ExprStatNode(node.result_expr.pos, expr=node.result_expr)
        else:
            body = Nodes.ReturnStatNode(node.result_expr.pos, value=node.result_expr)
        node.def_node = Nodes.DefNode(node.pos, name=node.name, args=node.args, star_arg=node.star_arg, starstar_arg=node.starstar_arg, body=body, doc=None)
        self.visitchildren(node)
        return node

    def visit_GeneratorExpressionNode(self, node):
        collector = YieldNodeCollector()
        collector.visitchildren(node.loop, attrs=None, exclude=['iterator'])
        node.def_node = Nodes.DefNode(node.pos, name=node.name, doc=None, args=[], star_arg=None, starstar_arg=None, body=node.loop, is_async_def=collector.has_await, is_generator_expression=True)
        _AssignmentExpressionChecker.do_checks(node.loop, scope_is_class=self.scope_type in ('pyclass', 'cclass'))
        self.visitchildren(node)
        return node

    def visit_ComprehensionNode(self, node):
        if not node.has_local_scope:
            collector = YieldNodeCollector()
            collector.visitchildren(node.loop)
            if collector.has_await:
                node.has_local_scope = True
        _AssignmentExpressionChecker.do_checks(node.loop, scope_is_class=self.scope_type in ('pyclass', 'cclass'))
        self.visitchildren(node)
        return node

    def handle_bufferdefaults(self, decl):
        if not isinstance(decl.default, ExprNodes.DictNode):
            raise PostParseError(decl.pos, ERR_BUF_DEFAULTS)
        self.scope_node.buffer_defaults_node = decl.default
        self.scope_node.buffer_defaults_pos = decl.pos

    def visit_CVarDefNode(self, node):
        try:
            self.visitchildren(node)
            stats = [node]
            newdecls = []
            for decl in node.declarators:
                declbase = decl
                while isinstance(declbase, Nodes.CPtrDeclaratorNode):
                    declbase = declbase.base
                if isinstance(declbase, Nodes.CNameDeclaratorNode):
                    if declbase.default is not None:
                        if self.scope_type in ('cclass', 'pyclass', 'struct'):
                            if isinstance(self.scope_node, Nodes.CClassDefNode):
                                handler = self.specialattribute_handlers.get(decl.name)
                                if handler:
                                    if decl is not declbase:
                                        raise PostParseError(decl.pos, ERR_INVALID_SPECIALATTR_TYPE)
                                    handler(decl)
                                    continue
                            raise PostParseError(decl.pos, ERR_CDEF_INCLASS)
                        first_assignment = self.scope_type != 'module'
                        stats.append(Nodes.SingleAssignmentNode(node.pos, lhs=ExprNodes.NameNode(node.pos, name=declbase.name), rhs=declbase.default, first=first_assignment))
                        declbase.default = None
                newdecls.append(decl)
            node.declarators = newdecls
            return stats
        except PostParseError as e:
            self.context.nonfatal_error(e)
            return None

    def visit_SingleAssignmentNode(self, node):
        self.visitchildren(node)
        return self._visit_assignment_node(node, [node.lhs, node.rhs])

    def visit_CascadedAssignmentNode(self, node):
        self.visitchildren(node)
        return self._visit_assignment_node(node, node.lhs_list + [node.rhs])

    def _visit_assignment_node(self, node, expr_list):
        """Flatten parallel assignments into separate single
        assignments or cascaded assignments.
        """
        if sum([1 for expr in expr_list if expr.is_sequence_constructor or expr.is_string_literal]) < 2:
            return node
        expr_list_list = []
        flatten_parallel_assignments(expr_list, expr_list_list)
        temp_refs = []
        eliminate_rhs_duplicates(expr_list_list, temp_refs)
        nodes = []
        for expr_list in expr_list_list:
            lhs_list = expr_list[:-1]
            rhs = expr_list[-1]
            if len(lhs_list) == 1:
                node = Nodes.SingleAssignmentNode(rhs.pos, lhs=lhs_list[0], rhs=rhs)
            else:
                node = Nodes.CascadedAssignmentNode(rhs.pos, lhs_list=lhs_list, rhs=rhs)
            nodes.append(node)
        if len(nodes) == 1:
            assign_node = nodes[0]
        else:
            assign_node = Nodes.ParallelAssignmentNode(nodes[0].pos, stats=nodes)
        if temp_refs:
            duplicates_and_temps = [(temp.expression, temp) for temp in temp_refs]
            sort_common_subsequences(duplicates_and_temps)
            for _, temp_ref in duplicates_and_temps[::-1]:
                assign_node = LetNode(temp_ref, assign_node)
        return assign_node

    def _flatten_sequence(self, seq, result):
        for arg in seq.args:
            if arg.is_sequence_constructor:
                self._flatten_sequence(arg, result)
            else:
                result.append(arg)
        return result

    def visit_DelStatNode(self, node):
        self.visitchildren(node)
        node.args = self._flatten_sequence(node, [])
        return node

    def visit_ExceptClauseNode(self, node):
        if node.is_except_as:
            del_target = Nodes.DelStatNode(node.pos, args=[ExprNodes.NameNode(node.target.pos, name=node.target.name)], ignore_nonexisting=True)
            node.body = Nodes.StatListNode(node.pos, stats=[Nodes.TryFinallyStatNode(node.pos, body=node.body, finally_clause=Nodes.StatListNode(node.pos, stats=[del_target]))])
        self.visitchildren(node)
        return node

    def visit_AssertStatNode(self, node):
        """Extract the exception raising into a RaiseStatNode to simplify GIL handling.
        """
        if node.exception is None:
            node.exception = Nodes.RaiseStatNode(node.pos, exc_type=ExprNodes.NameNode(node.pos, name=EncodedString('AssertionError')), exc_value=node.value, exc_tb=None, cause=None, builtin_exc_name='AssertionError', wrap_tuple_value=True)
            node.value = None
        self.visitchildren(node)
        return node