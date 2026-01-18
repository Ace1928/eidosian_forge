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
class NumBinopNode(BinopNode):
    infix = True
    overflow_check = False
    overflow_bit_node = None

    def analyse_c_operation(self, env):
        type1 = self.operand1.type
        type2 = self.operand2.type
        self.type = self.compute_c_result_type(type1, type2)
        if not self.type:
            self.type_error()
            return
        if self.type.is_complex:
            self.infix = False
        if self.type.is_int and env.directives['overflowcheck'] and (self.operator in self.overflow_op_names):
            if self.operator in ('+', '*') and self.operand1.has_constant_result() and (not self.operand2.has_constant_result()):
                self.operand1, self.operand2 = (self.operand2, self.operand1)
            self.overflow_check = True
            self.overflow_fold = env.directives['overflowcheck.fold']
            self.func = self.type.overflow_check_binop(self.overflow_op_names[self.operator], env, const_rhs=self.operand2.has_constant_result())
            self.is_temp = True
        if not self.infix or (type1.is_numeric and type2.is_numeric):
            self.operand1 = self.operand1.coerce_to(self.type, env)
            self.operand2 = self.operand2.coerce_to(self.type, env)

    def compute_c_result_type(self, type1, type2):
        if self.c_types_okay(type1, type2):
            widest_type = PyrexTypes.widest_numeric_type(type1, type2)
            if widest_type is PyrexTypes.c_bint_type:
                if self.operator not in '|^&':
                    widest_type = PyrexTypes.c_int_type
            else:
                widest_type = PyrexTypes.widest_numeric_type(widest_type, PyrexTypes.c_int_type)
            return widest_type
        else:
            return None

    def may_be_none(self):
        if self.type and self.type.is_builtin_type:
            return False
        type1 = self.operand1.type
        type2 = self.operand2.type
        if type1 and type1.is_builtin_type and type2 and type2.is_builtin_type:
            return False
        return super(NumBinopNode, self).may_be_none()

    def get_constant_c_result_code(self):
        value1 = self.operand1.get_constant_c_result_code()
        value2 = self.operand2.get_constant_c_result_code()
        if value1 and value2:
            return '(%s %s %s)' % (value1, self.operator, value2)
        else:
            return None

    def c_types_okay(self, type1, type2):
        return (type1.is_numeric or type1.is_enum) and (type2.is_numeric or type2.is_enum)

    def generate_evaluation_code(self, code):
        if self.overflow_check:
            self.overflow_bit_node = self
            self.overflow_bit = code.funcstate.allocate_temp(PyrexTypes.c_int_type, manage_ref=False)
            code.putln('%s = 0;' % self.overflow_bit)
        super(NumBinopNode, self).generate_evaluation_code(code)
        if self.overflow_check:
            code.putln('if (unlikely(%s)) {' % self.overflow_bit)
            code.putln('PyErr_SetString(PyExc_OverflowError, "value too large");')
            code.putln(code.error_goto(self.pos))
            code.putln('}')
            code.funcstate.release_temp(self.overflow_bit)

    def calculate_result_code(self):
        if self.overflow_bit_node is not None:
            return '%s(%s, %s, &%s)' % (self.func, self.operand1.result(), self.operand2.result(), self.overflow_bit_node.overflow_bit)
        elif self.type.is_cpp_class or self.infix:
            if is_pythran_expr(self.type):
                result1, result2 = (self.operand1.pythran_result(), self.operand2.pythran_result())
            else:
                result1, result2 = (self.operand1.result(), self.operand2.result())
            return '(%s %s %s)' % (result1, self.operator, result2)
        else:
            func = self.type.binary_op(self.operator)
            if func is None:
                error(self.pos, 'binary operator %s not supported for %s' % (self.operator, self.type))
            return '%s(%s, %s)' % (func, self.operand1.result(), self.operand2.result())

    def is_py_operation_types(self, type1, type2):
        return type1.is_unicode_char or type2.is_unicode_char or BinopNode.is_py_operation_types(self, type1, type2)

    def py_operation_function(self, code):
        function_name = self.py_functions[self.operator]
        if self.inplace:
            function_name = function_name.replace('PyNumber_', 'PyNumber_InPlace')
        return function_name
    py_functions = {'|': 'PyNumber_Or', '^': 'PyNumber_Xor', '&': 'PyNumber_And', '<<': 'PyNumber_Lshift', '>>': 'PyNumber_Rshift', '+': 'PyNumber_Add', '-': 'PyNumber_Subtract', '*': 'PyNumber_Multiply', '@': '__Pyx_PyNumber_MatrixMultiply', '/': '__Pyx_PyNumber_Divide', '//': 'PyNumber_FloorDivide', '%': 'PyNumber_Remainder', '**': 'PyNumber_Power'}
    overflow_op_names = {'+': 'add', '-': 'sub', '*': 'mul', '<<': 'lshift'}