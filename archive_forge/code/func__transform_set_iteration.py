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
def _transform_set_iteration(self, node, set_obj):
    temps = []
    temp = UtilNodes.TempHandle(PyrexTypes.py_object_type)
    temps.append(temp)
    set_temp = temp.ref(set_obj.pos)
    temp = UtilNodes.TempHandle(PyrexTypes.c_py_ssize_t_type)
    temps.append(temp)
    pos_temp = temp.ref(node.pos)
    if isinstance(node.body, Nodes.StatListNode):
        body = node.body
    else:
        body = Nodes.StatListNode(pos=node.body.pos, stats=[node.body])
    set_len_temp = UtilNodes.TempHandle(PyrexTypes.c_py_ssize_t_type)
    temps.append(set_len_temp)
    set_len_temp_addr = ExprNodes.AmpersandNode(node.pos, operand=set_len_temp.ref(set_obj.pos), type=PyrexTypes.c_ptr_type(set_len_temp.type))
    temp = UtilNodes.TempHandle(PyrexTypes.c_int_type)
    temps.append(temp)
    is_set_temp = temp.ref(node.pos)
    is_set_temp_addr = ExprNodes.AmpersandNode(node.pos, operand=is_set_temp, type=PyrexTypes.c_ptr_type(temp.type))
    value_target = node.target
    iter_next_node = Nodes.SetIterationNextNode(set_temp, set_len_temp.ref(set_obj.pos), pos_temp, value_target, is_set_temp)
    iter_next_node = iter_next_node.analyse_expressions(self.current_env())
    body.stats[0:0] = [iter_next_node]

    def flag_node(value):
        value = value and 1 or 0
        return ExprNodes.IntNode(node.pos, value=str(value), constant_result=value)
    result_code = [Nodes.SingleAssignmentNode(node.pos, lhs=pos_temp, rhs=ExprNodes.IntNode(node.pos, value='0', constant_result=0)), Nodes.SingleAssignmentNode(set_obj.pos, lhs=set_temp, rhs=ExprNodes.PythonCapiCallNode(set_obj.pos, '__Pyx_set_iterator', self.PySet_Iterator_func_type, utility_code=UtilityCode.load_cached('set_iter', 'Optimize.c'), args=[set_obj, flag_node(set_obj.type is Builtin.set_type), set_len_temp_addr, is_set_temp_addr], is_temp=True)), Nodes.WhileStatNode(node.pos, condition=None, body=body, else_clause=node.else_clause)]
    return UtilNodes.TempsBlockNode(node.pos, temps=temps, body=Nodes.StatListNode(node.pos, stats=result_code))