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
def _transform_unicode_iteration(self, node, slice_node, reversed=False):
    if slice_node.is_literal:
        try:
            bytes_value = bytes_literal(slice_node.value.encode('latin1'), 'iso8859-1')
        except UnicodeEncodeError:
            pass
        else:
            bytes_slice = ExprNodes.SliceIndexNode(slice_node.pos, base=ExprNodes.BytesNode(slice_node.pos, value=bytes_value, constant_result=bytes_value, type=PyrexTypes.c_const_char_ptr_type).coerce_to(PyrexTypes.c_const_uchar_ptr_type, self.current_env()), start=None, stop=ExprNodes.IntNode(slice_node.pos, value=str(len(bytes_value)), constant_result=len(bytes_value), type=PyrexTypes.c_py_ssize_t_type), type=Builtin.unicode_type)
            return self._transform_carray_iteration(node, bytes_slice, reversed)
    unpack_temp_node = UtilNodes.LetRefNode(slice_node.as_none_safe_node("'NoneType' is not iterable"))
    start_node = ExprNodes.IntNode(node.pos, value='0', constant_result=0, type=PyrexTypes.c_py_ssize_t_type)
    length_temp = UtilNodes.TempHandle(PyrexTypes.c_py_ssize_t_type)
    end_node = length_temp.ref(node.pos)
    if reversed:
        relation1, relation2 = ('>', '>=')
        start_node, end_node = (end_node, start_node)
    else:
        relation1, relation2 = ('<=', '<')
    kind_temp = UtilNodes.TempHandle(PyrexTypes.c_int_type)
    data_temp = UtilNodes.TempHandle(PyrexTypes.c_void_ptr_type)
    counter_temp = UtilNodes.TempHandle(PyrexTypes.c_py_ssize_t_type)
    target_value = ExprNodes.PythonCapiCallNode(slice_node.pos, '__Pyx_PyUnicode_READ', self.PyUnicode_READ_func_type, args=[kind_temp.ref(slice_node.pos), data_temp.ref(slice_node.pos), counter_temp.ref(node.target.pos)], is_temp=False)
    if target_value.type != node.target.type:
        target_value = target_value.coerce_to(node.target.type, self.current_env())
    target_assign = Nodes.SingleAssignmentNode(pos=node.target.pos, lhs=node.target, rhs=target_value)
    body = Nodes.StatListNode(node.pos, stats=[target_assign, node.body])
    loop_node = Nodes.ForFromStatNode(node.pos, bound1=start_node, relation1=relation1, target=counter_temp.ref(node.target.pos), relation2=relation2, bound2=end_node, step=None, body=body, else_clause=node.else_clause, from_range=True)
    setup_node = Nodes.ExprStatNode(node.pos, expr=ExprNodes.PythonCapiCallNode(slice_node.pos, '__Pyx_init_unicode_iteration', self.init_unicode_iteration_func_type, args=[unpack_temp_node, ExprNodes.AmpersandNode(slice_node.pos, operand=length_temp.ref(slice_node.pos), type=PyrexTypes.c_py_ssize_t_ptr_type), ExprNodes.AmpersandNode(slice_node.pos, operand=data_temp.ref(slice_node.pos), type=PyrexTypes.c_void_ptr_ptr_type), ExprNodes.AmpersandNode(slice_node.pos, operand=kind_temp.ref(slice_node.pos), type=PyrexTypes.c_int_ptr_type)], is_temp=True, result_is_used=False, utility_code=UtilityCode.load_cached('unicode_iter', 'Optimize.c')))
    return UtilNodes.LetNode(unpack_temp_node, UtilNodes.TempsBlockNode(node.pos, temps=[counter_temp, length_temp, data_temp, kind_temp], body=Nodes.StatListNode(node.pos, stats=[setup_node, loop_node])))