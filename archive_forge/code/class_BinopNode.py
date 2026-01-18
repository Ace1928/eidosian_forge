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
class BinopNode(ExprNode):
    subexprs = ['operand1', 'operand2']
    inplace = False

    def calculate_constant_result(self):
        func = compile_time_binary_operators[self.operator]
        self.constant_result = func(self.operand1.constant_result, self.operand2.constant_result)

    def compile_time_value(self, denv):
        func = get_compile_time_binop(self)
        operand1 = self.operand1.compile_time_value(denv)
        operand2 = self.operand2.compile_time_value(denv)
        try:
            return func(operand1, operand2)
        except Exception as e:
            self.compile_time_value_error(e)

    def infer_type(self, env):
        return self.result_type(self.operand1.infer_type(env), self.operand2.infer_type(env), env)

    def analyse_types(self, env):
        self.operand1 = self.operand1.analyse_types(env)
        self.operand2 = self.operand2.analyse_types(env)
        self.analyse_operation(env)
        return self

    def analyse_operation(self, env):
        if self.is_pythran_operation(env):
            self.type = self.result_type(self.operand1.type, self.operand2.type, env)
            assert self.type.is_pythran_expr
            self.is_temp = 1
        elif self.is_py_operation():
            self.coerce_operands_to_pyobjects(env)
            self.type = self.result_type(self.operand1.type, self.operand2.type, env)
            assert self.type.is_pyobject
            self.is_temp = 1
        elif self.is_cpp_operation():
            self.analyse_cpp_operation(env)
        else:
            self.analyse_c_operation(env)

    def is_py_operation(self):
        return self.is_py_operation_types(self.operand1.type, self.operand2.type)

    def is_py_operation_types(self, type1, type2):
        return type1.is_pyobject or type2.is_pyobject or type1.is_ctuple or type2.is_ctuple

    def is_pythran_operation(self, env):
        return self.is_pythran_operation_types(self.operand1.type, self.operand2.type, env)

    def is_pythran_operation_types(self, type1, type2, env):
        return has_np_pythran(env) and (is_pythran_supported_operation_type(type1) and is_pythran_supported_operation_type(type2)) and (is_pythran_expr(type1) or is_pythran_expr(type2))

    def is_cpp_operation(self):
        return self.operand1.type.is_cpp_class or self.operand2.type.is_cpp_class

    def analyse_cpp_operation(self, env):
        entry = env.lookup_operator(self.operator, [self.operand1, self.operand2])
        if not entry:
            self.type_error()
            return
        func_type = entry.type
        self.exception_check = func_type.exception_check
        self.exception_value = func_type.exception_value
        if self.exception_check == '+':
            self.is_temp = 1
            if needs_cpp_exception_conversion(self):
                env.use_utility_code(UtilityCode.load_cached('CppExceptionConversion', 'CppSupport.cpp'))
        if func_type.is_ptr:
            func_type = func_type.base_type
        if len(func_type.args) == 1:
            self.operand2 = self.operand2.coerce_to(func_type.args[0].type, env)
        else:
            self.operand1 = self.operand1.coerce_to(func_type.args[0].type, env)
            self.operand2 = self.operand2.coerce_to(func_type.args[1].type, env)
        self.type = func_type.return_type

    def result_type(self, type1, type2, env):
        if self.is_pythran_operation_types(type1, type2, env):
            return PythranExpr(pythran_binop_type(self.operator, type1, type2))
        if self.is_py_operation_types(type1, type2):
            if type2.is_string:
                type2 = Builtin.bytes_type
            elif type2.is_pyunicode_ptr:
                type2 = Builtin.unicode_type
            if type1.is_string:
                type1 = Builtin.bytes_type
            elif type1.is_pyunicode_ptr:
                type1 = Builtin.unicode_type
            if type1.is_builtin_type or type2.is_builtin_type:
                if type1 is type2 and self.operator in '**%+|&^':
                    return type1
                result_type = self.infer_builtin_types_operation(type1, type2)
                if result_type is not None:
                    return result_type
            return py_object_type
        elif type1.is_error or type2.is_error:
            return PyrexTypes.error_type
        else:
            return self.compute_c_result_type(type1, type2)

    def infer_builtin_types_operation(self, type1, type2):
        return None

    def nogil_check(self, env):
        if self.is_py_operation():
            self.gil_error()

    def coerce_operands_to_pyobjects(self, env):
        self.operand1 = self.operand1.coerce_to_pyobject(env)
        self.operand2 = self.operand2.coerce_to_pyobject(env)

    def check_const(self):
        return self.operand1.check_const() and self.operand2.check_const()

    def is_ephemeral(self):
        return super(BinopNode, self).is_ephemeral() or self.operand1.is_ephemeral() or self.operand2.is_ephemeral()

    def generate_result_code(self, code):
        type1 = self.operand1.type
        type2 = self.operand2.type
        if self.type.is_pythran_expr:
            code.putln('// Pythran binop')
            code.putln('__Pyx_call_destructor(%s);' % self.result())
            if self.operator == '**':
                code.putln('new (&%s) decltype(%s){pythonic::numpy::functor::power{}(%s, %s)};' % (self.result(), self.result(), self.operand1.pythran_result(), self.operand2.pythran_result()))
            else:
                code.putln('new (&%s) decltype(%s){%s %s %s};' % (self.result(), self.result(), self.operand1.pythran_result(), self.operator, self.operand2.pythran_result()))
        elif type1.is_pyobject or type2.is_pyobject:
            function = self.py_operation_function(code)
            extra_args = ', Py_None' if self.operator == '**' else ''
            op1_result = self.operand1.py_result() if type1.is_pyobject else self.operand1.result()
            op2_result = self.operand2.py_result() if type2.is_pyobject else self.operand2.result()
            code.putln('%s = %s(%s, %s%s); %s' % (self.result(), function, op1_result, op2_result, extra_args, code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
        elif self.is_temp:
            if self.is_cpp_operation() and self.exception_check == '+':
                translate_cpp_exception(code, self.pos, '%s = %s;' % (self.result(), self.calculate_result_code()), self.result() if self.type.is_pyobject else None, self.exception_value, self.in_nogil_context)
            else:
                code.putln('%s = %s;' % (self.result(), self.calculate_result_code()))

    def type_error(self):
        if not (self.operand1.type.is_error or self.operand2.type.is_error):
            error(self.pos, "Invalid operand types for '%s' (%s; %s)" % (self.operator, self.operand1.type, self.operand2.type))
        self.type = PyrexTypes.error_type