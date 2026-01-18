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
class DivNode(NumBinopNode):
    cdivision = None
    truedivision = None
    ctruedivision = False
    cdivision_warnings = False
    zerodivision_check = None

    def find_compile_time_binary_operator(self, op1, op2):
        func = compile_time_binary_operators[self.operator]
        if self.operator == '/' and self.truedivision is None:
            if isinstance(op1, _py_int_types) and isinstance(op2, _py_int_types):
                func = compile_time_binary_operators['//']
        return func

    def calculate_constant_result(self):
        op1 = self.operand1.constant_result
        op2 = self.operand2.constant_result
        func = self.find_compile_time_binary_operator(op1, op2)
        self.constant_result = func(self.operand1.constant_result, self.operand2.constant_result)

    def compile_time_value(self, denv):
        operand1 = self.operand1.compile_time_value(denv)
        operand2 = self.operand2.compile_time_value(denv)
        try:
            func = self.find_compile_time_binary_operator(operand1, operand2)
            return func(operand1, operand2)
        except Exception as e:
            self.compile_time_value_error(e)

    def _check_truedivision(self, env):
        if self.cdivision or env.directives['cdivision']:
            self.ctruedivision = False
        else:
            self.ctruedivision = self.truedivision

    def infer_type(self, env):
        self._check_truedivision(env)
        return self.result_type(self.operand1.infer_type(env), self.operand2.infer_type(env), env)

    def analyse_operation(self, env):
        self._check_truedivision(env)
        NumBinopNode.analyse_operation(self, env)
        if self.is_cpp_operation():
            self.cdivision = True
        if not self.type.is_pyobject:
            self.zerodivision_check = self.cdivision is None and (not env.directives['cdivision']) and (not self.operand2.has_constant_result() or self.operand2.constant_result == 0)
            if self.zerodivision_check or env.directives['cdivision_warnings']:
                self.operand1 = self.operand1.coerce_to_simple(env)
                self.operand2 = self.operand2.coerce_to_simple(env)

    def compute_c_result_type(self, type1, type2):
        if self.operator == '/' and self.ctruedivision and (not type1.is_cpp_class) and (not type2.is_cpp_class):
            if not type1.is_float and (not type2.is_float):
                widest_type = PyrexTypes.widest_numeric_type(type1, PyrexTypes.c_double_type)
                widest_type = PyrexTypes.widest_numeric_type(type2, widest_type)
                return widest_type
        return NumBinopNode.compute_c_result_type(self, type1, type2)

    def zero_division_message(self):
        if self.type.is_int:
            return 'integer division or modulo by zero'
        else:
            return 'float division'

    def generate_evaluation_code(self, code):
        if not self.type.is_pyobject and (not self.type.is_complex):
            if self.cdivision is None:
                self.cdivision = code.globalstate.directives['cdivision'] or self.type.is_float or ((self.type.is_numeric or self.type.is_enum) and (not self.type.signed))
            if not self.cdivision:
                code.globalstate.use_utility_code(UtilityCode.load_cached('DivInt', 'CMath.c').specialize(self.type))
        NumBinopNode.generate_evaluation_code(self, code)
        self.generate_div_warning_code(code)

    def generate_div_warning_code(self, code):
        in_nogil = self.in_nogil_context
        if not self.type.is_pyobject:
            if self.zerodivision_check:
                if not self.infix:
                    zero_test = '%s(%s)' % (self.type.unary_op('zero'), self.operand2.result())
                else:
                    zero_test = '%s == 0' % self.operand2.result()
                code.putln('if (unlikely(%s)) {' % zero_test)
                if in_nogil:
                    code.put_ensure_gil()
                code.putln('PyErr_SetString(PyExc_ZeroDivisionError, "%s");' % self.zero_division_message())
                if in_nogil:
                    code.put_release_ensured_gil()
                code.putln(code.error_goto(self.pos))
                code.putln('}')
                if self.type.is_int and self.type.signed and (self.operator != '%'):
                    code.globalstate.use_utility_code(UtilityCode.load_cached('UnaryNegOverflows', 'Overflow.c'))
                    if self.operand2.type.signed == 2:
                        minus1_check = 'unlikely(%s == -1)' % self.operand2.result()
                    else:
                        type_of_op2 = self.operand2.type.empty_declaration_code()
                        minus1_check = '(!(((%s)-1) > 0)) && unlikely(%s == (%s)-1)' % (type_of_op2, self.operand2.result(), type_of_op2)
                    code.putln('else if (sizeof(%s) == sizeof(long) && %s  && unlikely(__Pyx_UNARY_NEG_WOULD_OVERFLOW(%s))) {' % (self.type.empty_declaration_code(), minus1_check, self.operand1.result()))
                    if in_nogil:
                        code.put_ensure_gil()
                    code.putln('PyErr_SetString(PyExc_OverflowError, "value too large to perform division");')
                    if in_nogil:
                        code.put_release_ensured_gil()
                    code.putln(code.error_goto(self.pos))
                    code.putln('}')
            if code.globalstate.directives['cdivision_warnings'] and self.operator != '/':
                code.globalstate.use_utility_code(UtilityCode.load_cached('CDivisionWarning', 'CMath.c'))
                code.putln('if (unlikely((%s < 0) ^ (%s < 0))) {' % (self.operand1.result(), self.operand2.result()))
                warning_code = '__Pyx_cdivision_warning(%(FILENAME)s, %(LINENO)s)' % {'FILENAME': Naming.filename_cname, 'LINENO': Naming.lineno_cname}
                if in_nogil:
                    result_code = 'result'
                    code.putln('int %s;' % result_code)
                    code.put_ensure_gil()
                    code.putln(code.set_error_info(self.pos, used=True))
                    code.putln('%s = %s;' % (result_code, warning_code))
                    code.put_release_ensured_gil()
                else:
                    result_code = warning_code
                    code.putln(code.set_error_info(self.pos, used=True))
                code.put('if (unlikely(%s)) ' % result_code)
                code.put_goto(code.error_label)
                code.putln('}')

    def calculate_result_code(self):
        if self.type.is_complex or self.is_cpp_operation():
            return NumBinopNode.calculate_result_code(self)
        elif self.type.is_float and self.operator == '//':
            return 'floor(%s / %s)' % (self.operand1.result(), self.operand2.result())
        elif self.truedivision or self.cdivision:
            op1 = self.operand1.result()
            op2 = self.operand2.result()
            if self.truedivision:
                if self.type != self.operand1.type:
                    op1 = self.type.cast_code(op1)
                if self.type != self.operand2.type:
                    op2 = self.type.cast_code(op2)
            return '(%s / %s)' % (op1, op2)
        else:
            return '__Pyx_div_%s(%s, %s)' % (self.type.specialization_name(), self.operand1.result(), self.operand2.result())