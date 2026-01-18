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
def _handle_simple_method_list_extend(self, node, function, args, is_unbound_method):
    """Replace list.extend([...]) for short sequence literals values by sequential appends
        to avoid creating an intermediate sequence argument.
        """
    if len(args) != 2:
        return node
    obj, value = args
    if not value.is_sequence_constructor:
        return node
    items = list(value.args)
    if value.mult_factor is not None or len(items) > 8:
        if False and isinstance(value, ExprNodes.ListNode):
            tuple_node = args[1].as_tuple().analyse_types(self.current_env(), skip_children=True)
            Visitor.recursively_replace_node(node, args[1], tuple_node)
        return node
    wrapped_obj = self._wrap_self_arg(obj, function, is_unbound_method, 'extend')
    if not items:
        wrapped_obj.result_is_used = node.result_is_used
        return wrapped_obj
    cloned_obj = obj = wrapped_obj
    if len(items) > 1 and (not obj.is_simple()):
        cloned_obj = UtilNodes.LetRefNode(obj)
    temps = []
    arg = items[-1]
    if not arg.is_simple():
        arg = UtilNodes.LetRefNode(arg)
        temps.append(arg)
    new_node = ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_PyList_Append', self.PyObject_Append_func_type, args=[cloned_obj, arg], is_temp=True, utility_code=load_c_utility('ListAppend'))
    for arg in items[-2::-1]:
        if not arg.is_simple():
            arg = UtilNodes.LetRefNode(arg)
            temps.append(arg)
        new_node = ExprNodes.binop_node(node.pos, '|', ExprNodes.PythonCapiCallNode(node.pos, '__Pyx_ListComp_Append', self.PyObject_Append_func_type, args=[cloned_obj, arg], py_name='extend', is_temp=True, utility_code=load_c_utility('ListCompAppend')), new_node, type=PyrexTypes.c_returncode_type)
    new_node.result_is_used = node.result_is_used
    if cloned_obj is not obj:
        temps.append(cloned_obj)
    for temp in temps:
        new_node = UtilNodes.EvalWithTempExprNode(temp, new_node)
        new_node.result_is_used = node.result_is_used
    return new_node