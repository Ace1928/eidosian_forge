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
class PrimaryCmpNode(ExprNode, CmpNode):
    child_attrs = ['operand1', 'operand2', 'coerced_operand2', 'cascade', 'special_bool_extra_args']
    cascade = None
    coerced_operand2 = None
    is_memslice_nonecheck = False

    def infer_type(self, env):
        type1 = self.operand1.infer_type(env)
        type2 = self.operand2.infer_type(env)
        if is_pythran_expr(type1) or is_pythran_expr(type2):
            if is_pythran_supported_type(type1) and is_pythran_supported_type(type2):
                return PythranExpr(pythran_binop_type(self.operator, type1, type2))
        return py_object_type

    def type_dependencies(self, env):
        return ()

    def calculate_constant_result(self):
        assert not self.cascade
        self.calculate_cascaded_constant_result(self.operand1.constant_result)

    def compile_time_value(self, denv):
        operand1 = self.operand1.compile_time_value(denv)
        return self.cascaded_compile_time_value(operand1, denv)

    def unify_cascade_type(self):
        cdr = self.cascade
        while cdr:
            cdr.type = self.type
            cdr = cdr.cascade

    def analyse_types(self, env):
        self.operand1 = self.operand1.analyse_types(env)
        self.operand2 = self.operand2.analyse_types(env)
        if self.is_cpp_comparison():
            self.analyse_cpp_comparison(env)
            if self.cascade:
                error(self.pos, 'Cascading comparison not yet supported for cpp types.')
            return self
        type1 = self.operand1.type
        type2 = self.operand2.type
        if is_pythran_expr(type1) or is_pythran_expr(type2):
            if is_pythran_supported_type(type1) and is_pythran_supported_type(type2):
                self.type = PythranExpr(pythran_binop_type(self.operator, type1, type2))
                self.is_pycmp = False
                return self
        if self.analyse_memoryviewslice_comparison(env):
            return self
        if self.cascade:
            self.cascade = self.cascade.analyse_types(env)
        if self.operator in ('in', 'not_in'):
            if self.is_c_string_contains():
                self.is_pycmp = False
                common_type = None
                if self.cascade:
                    error(self.pos, "Cascading comparison not yet supported for 'int_val in string'.")
                    return self
                if self.operand2.type is unicode_type:
                    env.use_utility_code(UtilityCode.load_cached('PyUCS4InUnicode', 'StringTools.c'))
                else:
                    if self.operand1.type is PyrexTypes.c_uchar_type:
                        self.operand1 = self.operand1.coerce_to(PyrexTypes.c_char_type, env)
                    if self.operand2.type is not bytes_type:
                        self.operand2 = self.operand2.coerce_to(bytes_type, env)
                    env.use_utility_code(UtilityCode.load_cached('BytesContains', 'StringTools.c'))
                self.operand2 = self.operand2.as_none_safe_node("argument of type 'NoneType' is not iterable")
            elif self.is_ptr_contains():
                if self.cascade:
                    error(self.pos, "Cascading comparison not supported for 'val in sliced pointer'.")
                self.type = PyrexTypes.c_bint_type
                return self
            elif self.find_special_bool_compare_function(env, self.operand1):
                if not self.operand1.type.is_pyobject:
                    self.operand1 = self.operand1.coerce_to_pyobject(env)
                common_type = None
                self.is_pycmp = False
            else:
                common_type = py_object_type
                self.is_pycmp = True
        elif self.find_special_bool_compare_function(env, self.operand1):
            if not self.operand1.type.is_pyobject:
                self.operand1 = self.operand1.coerce_to_pyobject(env)
            common_type = None
            self.is_pycmp = False
        else:
            common_type = self.find_common_type(env, self.operator, self.operand1)
            self.is_pycmp = common_type.is_pyobject
        if common_type is not None and (not common_type.is_error):
            if self.operand1.type != common_type:
                self.operand1 = self.operand1.coerce_to(common_type, env)
            self.coerce_operands_to(common_type, env)
        if self.cascade:
            self.operand2 = self.operand2.coerce_to_simple(env)
            self.cascade.coerce_cascaded_operands_to_temp(env)
            operand2 = self.cascade.optimise_comparison(self.operand2, env)
            if operand2 is not self.operand2:
                self.coerced_operand2 = operand2
        if self.is_python_result():
            self.type = PyrexTypes.py_object_type
        else:
            self.type = PyrexTypes.c_bint_type
        self.unify_cascade_type()
        if self.is_pycmp or self.cascade or self.special_bool_cmp_function:
            self.is_temp = 1
        return self

    def analyse_cpp_comparison(self, env):
        type1 = self.operand1.type
        type2 = self.operand2.type
        self.is_pycmp = False
        entry = env.lookup_operator(self.operator, [self.operand1, self.operand2])
        if entry is None:
            error(self.pos, "Invalid types for '%s' (%s, %s)" % (self.operator, type1, type2))
            self.type = PyrexTypes.error_type
            self.result_code = '<error>'
            return
        func_type = entry.type
        if func_type.is_ptr:
            func_type = func_type.base_type
        self.exception_check = func_type.exception_check
        self.exception_value = func_type.exception_value
        if self.exception_check == '+':
            self.is_temp = True
            if needs_cpp_exception_conversion(self):
                env.use_utility_code(UtilityCode.load_cached('CppExceptionConversion', 'CppSupport.cpp'))
        if len(func_type.args) == 1:
            self.operand2 = self.operand2.coerce_to(func_type.args[0].type, env)
        else:
            self.operand1 = self.operand1.coerce_to(func_type.args[0].type, env)
            self.operand2 = self.operand2.coerce_to(func_type.args[1].type, env)
        self.type = func_type.return_type

    def analyse_memoryviewslice_comparison(self, env):
        have_none = self.operand1.is_none or self.operand2.is_none
        have_slice = self.operand1.type.is_memoryviewslice or self.operand2.type.is_memoryviewslice
        ops = ('==', '!=', 'is', 'is_not')
        if have_slice and have_none and (self.operator in ops):
            self.is_pycmp = False
            self.type = PyrexTypes.c_bint_type
            self.is_memslice_nonecheck = True
            return True
        return False

    def coerce_to_boolean(self, env):
        if self.is_pycmp:
            if self.find_special_bool_compare_function(env, self.operand1, result_is_bool=True):
                self.is_pycmp = False
                self.type = PyrexTypes.c_bint_type
                self.is_temp = 1
                if self.cascade:
                    operand2 = self.cascade.optimise_comparison(self.operand2, env, result_is_bool=True)
                    if operand2 is not self.operand2:
                        self.coerced_operand2 = operand2
                self.unify_cascade_type()
                return self
        return ExprNode.coerce_to_boolean(self, env)

    def has_python_operands(self):
        return self.operand1.type.is_pyobject or self.operand2.type.is_pyobject

    def check_const(self):
        if self.cascade:
            self.not_const()
            return False
        else:
            return self.operand1.check_const() and self.operand2.check_const()

    def calculate_result_code(self):
        operand1, operand2 = (self.operand1, self.operand2)
        if operand1.type.is_complex:
            if self.operator == '!=':
                negation = '!'
            else:
                negation = ''
            return '(%s%s(%s, %s))' % (negation, operand1.type.binary_op('=='), operand1.result(), operand2.result())
        elif self.is_c_string_contains():
            if operand2.type is unicode_type:
                method = '__Pyx_UnicodeContainsUCS4'
            else:
                method = '__Pyx_BytesContains'
            if self.operator == 'not_in':
                negation = '!'
            else:
                negation = ''
            return '(%s%s(%s, %s))' % (negation, method, operand2.result(), operand1.result())
        else:
            if is_pythran_expr(self.type):
                result1, result2 = (operand1.pythran_result(), operand2.pythran_result())
            else:
                result1, result2 = (operand1.result(), operand2.result())
                if self.is_memslice_nonecheck:
                    if operand1.type.is_memoryviewslice:
                        result1 = '((PyObject *) %s.memview)' % result1
                    else:
                        result2 = '((PyObject *) %s.memview)' % result2
            return '(%s %s %s)' % (result1, self.c_operator(self.operator), result2)

    def generate_evaluation_code(self, code):
        self.operand1.generate_evaluation_code(code)
        self.operand2.generate_evaluation_code(code)
        for extra_arg in self.special_bool_extra_args:
            extra_arg.generate_evaluation_code(code)
        if self.is_temp:
            self.allocate_temp_result(code)
            self.generate_operation_code(code, self.result(), self.operand1, self.operator, self.operand2)
            if self.cascade:
                self.cascade.generate_evaluation_code(code, self.result(), self.coerced_operand2 or self.operand2, needs_evaluation=self.coerced_operand2 is not None)
            self.operand1.generate_disposal_code(code)
            self.operand1.free_temps(code)
            self.operand2.generate_disposal_code(code)
            self.operand2.free_temps(code)

    def generate_subexpr_disposal_code(self, code):
        self.operand1.generate_disposal_code(code)
        self.operand2.generate_disposal_code(code)

    def free_subexpr_temps(self, code):
        self.operand1.free_temps(code)
        self.operand2.free_temps(code)

    def annotate(self, code):
        self.operand1.annotate(code)
        self.operand2.annotate(code)
        if self.cascade:
            self.cascade.annotate(code)