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