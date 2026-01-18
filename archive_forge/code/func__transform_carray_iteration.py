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
def _transform_carray_iteration(self, node, slice_node, reversed=False):
    neg_step = False
    if isinstance(slice_node, ExprNodes.SliceIndexNode):
        slice_base = slice_node.base
        start = filter_none_node(slice_node.start)
        stop = filter_none_node(slice_node.stop)
        step = None
        if not stop:
            if not slice_base.type.is_pyobject:
                error(slice_node.pos, 'C array iteration requires known end index')
            return node
    elif slice_node.is_subscript:
        assert isinstance(slice_node.index, ExprNodes.SliceNode)
        slice_base = slice_node.base
        index = slice_node.index
        start = filter_none_node(index.start)
        stop = filter_none_node(index.stop)
        step = filter_none_node(index.step)
        if step:
            if not isinstance(step.constant_result, _py_int_types) or step.constant_result == 0 or (step.constant_result > 0 and (not stop)) or (step.constant_result < 0 and (not start)):
                if not slice_base.type.is_pyobject:
                    error(step.pos, 'C array iteration requires known step size and end index')
                return node
            else:
                step_value = step.constant_result
                if reversed:
                    step_value = -step_value
                neg_step = step_value < 0
                step = ExprNodes.IntNode(step.pos, type=PyrexTypes.c_py_ssize_t_type, value=str(abs(step_value)), constant_result=abs(step_value))
    elif slice_node.type.is_array:
        if slice_node.type.size is None:
            error(slice_node.pos, 'C array iteration requires known end index')
            return node
        slice_base = slice_node
        start = None
        stop = ExprNodes.IntNode(slice_node.pos, value=str(slice_node.type.size), type=PyrexTypes.c_py_ssize_t_type, constant_result=slice_node.type.size)
        step = None
    else:
        if not slice_node.type.is_pyobject:
            error(slice_node.pos, 'C array iteration requires known end index')
        return node
    if start:
        start = start.coerce_to(PyrexTypes.c_py_ssize_t_type, self.current_env())
    if stop:
        stop = stop.coerce_to(PyrexTypes.c_py_ssize_t_type, self.current_env())
    if stop is None:
        if neg_step:
            stop = ExprNodes.IntNode(slice_node.pos, value='-1', type=PyrexTypes.c_py_ssize_t_type, constant_result=-1)
        else:
            error(slice_node.pos, 'C array iteration requires known step size and end index')
            return node
    if reversed:
        if not start:
            start = ExprNodes.IntNode(slice_node.pos, value='0', constant_result=0, type=PyrexTypes.c_py_ssize_t_type)
        start, stop = (stop, start)
    ptr_type = slice_base.type
    if ptr_type.is_array:
        ptr_type = ptr_type.element_ptr_type()
    carray_ptr = slice_base.coerce_to_simple(self.current_env())
    if start and start.constant_result != 0:
        start_ptr_node = ExprNodes.AddNode(start.pos, operand1=carray_ptr, operator='+', operand2=start, type=ptr_type)
    else:
        start_ptr_node = carray_ptr
    if stop and stop.constant_result != 0:
        stop_ptr_node = ExprNodes.AddNode(stop.pos, operand1=ExprNodes.CloneNode(carray_ptr), operator='+', operand2=stop, type=ptr_type).coerce_to_simple(self.current_env())
    else:
        stop_ptr_node = ExprNodes.CloneNode(carray_ptr)
    counter = UtilNodes.TempHandle(ptr_type)
    counter_temp = counter.ref(node.target.pos)
    if slice_base.type.is_string and node.target.type.is_pyobject:
        if slice_node.type is Builtin.unicode_type:
            target_value = ExprNodes.CastNode(ExprNodes.DereferenceNode(node.target.pos, operand=counter_temp, type=ptr_type.base_type), PyrexTypes.c_py_ucs4_type).coerce_to(node.target.type, self.current_env())
        else:
            target_value = ExprNodes.SliceIndexNode(node.target.pos, start=ExprNodes.IntNode(node.target.pos, value='0', constant_result=0, type=PyrexTypes.c_int_type), stop=ExprNodes.IntNode(node.target.pos, value='1', constant_result=1, type=PyrexTypes.c_int_type), base=counter_temp, type=Builtin.bytes_type, is_temp=1)
    elif node.target.type.is_ptr and (not node.target.type.assignable_from(ptr_type.base_type)):
        target_value = counter_temp
    else:
        target_value = ExprNodes.IndexNode(node.target.pos, index=ExprNodes.IntNode(node.target.pos, value='0', constant_result=0, type=PyrexTypes.c_int_type), base=counter_temp, type=ptr_type.base_type)
    if target_value.type != node.target.type:
        target_value = target_value.coerce_to(node.target.type, self.current_env())
    target_assign = Nodes.SingleAssignmentNode(pos=node.target.pos, lhs=node.target, rhs=target_value)
    body = Nodes.StatListNode(node.pos, stats=[target_assign, node.body])
    relation1, relation2 = self._find_for_from_node_relations(neg_step, reversed)
    for_node = Nodes.ForFromStatNode(node.pos, bound1=start_ptr_node, relation1=relation1, target=counter_temp, relation2=relation2, bound2=stop_ptr_node, step=step, body=body, else_clause=node.else_clause, from_range=True)
    return UtilNodes.TempsBlockNode(node.pos, temps=[counter], body=for_node)