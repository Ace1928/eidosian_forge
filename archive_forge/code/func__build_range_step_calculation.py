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
def _build_range_step_calculation(self, bound1, bound2_ref_node, step, step_value):
    abs_step = abs(step_value)
    spanning_type = PyrexTypes.spanning_type(bound1.type, bound2_ref_node.type)
    if step.type.is_int and abs_step < 32767:
        spanning_step_type = PyrexTypes.spanning_type(spanning_type, PyrexTypes.c_int_type)
    else:
        spanning_step_type = PyrexTypes.spanning_type(spanning_type, step.type)
    if step_value < 0:
        begin_value = bound2_ref_node
        end_value = bound1
        final_op = '-'
    else:
        begin_value = bound1
        end_value = bound2_ref_node
        final_op = '+'
    step_calculation_node = ExprNodes.binop_node(bound1.pos, operand1=ExprNodes.binop_node(bound1.pos, operand1=bound2_ref_node, operator=final_op, operand2=ExprNodes.MulNode(bound1.pos, operand1=ExprNodes.IntNode(bound1.pos, value=str(abs_step), constant_result=abs_step, type=spanning_step_type), operator='*', operand2=ExprNodes.DivNode(bound1.pos, operand1=ExprNodes.SubNode(bound1.pos, operand1=ExprNodes.SubNode(bound1.pos, operand1=begin_value, operator='-', operand2=end_value, type=spanning_type), operator='-', operand2=ExprNodes.IntNode(bound1.pos, value='1', constant_result=1), type=spanning_step_type), operator='//', operand2=ExprNodes.IntNode(bound1.pos, value=str(abs_step), constant_result=abs_step, type=spanning_step_type), type=spanning_step_type), type=spanning_step_type), type=spanning_step_type), operator=final_op, operand2=ExprNodes.IntNode(bound1.pos, value='1', constant_result=1), type=spanning_type)
    return step_calculation_node