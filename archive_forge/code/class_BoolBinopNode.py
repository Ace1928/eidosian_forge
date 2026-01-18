from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
class BoolBinopNode(ExprNode):
    """
    Short-circuiting boolean operation.

    Note that this node provides the same code generation method as
    BoolBinopResultNode to simplify expression nesting.

    operator  string                              "and"/"or"
    operand1  BoolBinopNode/BoolBinopResultNode   left operand
    operand2  BoolBinopNode/BoolBinopResultNode   right operand
    """
    subexprs = ['operand1', 'operand2']
    is_temp = True
    operator = None
    operand1 = None
    operand2 = None

    def infer_type(self, env):
        type1 = self.operand1.infer_type(env)
        type2 = self.operand2.infer_type(env)
        return PyrexTypes.independent_spanning_type(type1, type2)

    def may_be_none(self):
        if self.operator == 'or':
            return self.operand2.may_be_none()
        else:
            return self.operand1.may_be_none() or self.operand2.may_be_none()

    def calculate_constant_result(self):
        operand1 = self.operand1.constant_result
        operand2 = self.operand2.constant_result
        if self.operator == 'and':
            self.constant_result = operand1 and operand2
        else:
            self.constant_result = operand1 or operand2

    def compile_time_value(self, denv):
        operand1 = self.operand1.compile_time_value(denv)
        operand2 = self.operand2.compile_time_value(denv)
        if self.operator == 'and':
            return operand1 and operand2
        else:
            return operand1 or operand2

    def is_ephemeral(self):
        return self.operand1.is_ephemeral() or self.operand2.is_ephemeral()

    def analyse_types(self, env):
        operand1 = self.operand1.analyse_types(env)
        operand2 = self.operand2.analyse_types(env)
        self.type = PyrexTypes.independent_spanning_type(operand1.type, operand2.type)
        self.operand1 = self._wrap_operand(operand1, env)
        self.operand2 = self._wrap_operand(operand2, env)
        return self

    def _wrap_operand(self, operand, env):
        if not isinstance(operand, (BoolBinopNode, BoolBinopResultNode)):
            operand = BoolBinopResultNode(operand, self.type, env)
        return operand

    def wrap_operands(self, env):
        """
        Must get called by transforms that want to create a correct BoolBinopNode
        after the type analysis phase.
        """
        self.operand1 = self._wrap_operand(self.operand1, env)
        self.operand2 = self._wrap_operand(self.operand2, env)

    def coerce_to_boolean(self, env):
        return self.coerce_to(PyrexTypes.c_bint_type, env)

    def coerce_to(self, dst_type, env):
        operand1 = self.operand1.coerce_to(dst_type, env)
        operand2 = self.operand2.coerce_to(dst_type, env)
        return BoolBinopNode.from_node(self, type=dst_type, operator=self.operator, operand1=operand1, operand2=operand2)

    def generate_bool_evaluation_code(self, code, final_result_temp, final_result_type, and_label, or_label, end_label, fall_through):
        code.mark_pos(self.pos)
        outer_labels = (and_label, or_label)
        if self.operator == 'and':
            my_label = and_label = code.new_label('next_and')
        else:
            my_label = or_label = code.new_label('next_or')
        self.operand1.generate_bool_evaluation_code(code, final_result_temp, final_result_type, and_label, or_label, end_label, my_label)
        and_label, or_label = outer_labels
        code.put_label(my_label)
        self.operand2.generate_bool_evaluation_code(code, final_result_temp, final_result_type, and_label, or_label, end_label, fall_through)

    def generate_evaluation_code(self, code):
        self.allocate_temp_result(code)
        result_type = PyrexTypes.py_object_type if self.type.is_pyobject else self.type
        or_label = and_label = None
        end_label = code.new_label('bool_binop_done')
        self.generate_bool_evaluation_code(code, self.result(), result_type, and_label, or_label, end_label, end_label)
        code.put_label(end_label)
    gil_message = 'Truth-testing Python object'

    def check_const(self):
        return self.operand1.check_const() and self.operand2.check_const()

    def generate_subexpr_disposal_code(self, code):
        pass

    def free_subexpr_temps(self, code):
        pass

    def generate_operand1_test(self, code):
        if self.type.is_pyobject:
            test_result = code.funcstate.allocate_temp(PyrexTypes.c_bint_type, manage_ref=False)
            code.putln('%s = __Pyx_PyObject_IsTrue(%s); %s' % (test_result, self.operand1.py_result(), code.error_goto_if_neg(test_result, self.pos)))
        else:
            test_result = self.operand1.result()
        return (test_result, self.type.is_pyobject)