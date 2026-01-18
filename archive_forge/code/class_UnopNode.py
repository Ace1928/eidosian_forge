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
class UnopNode(ExprNode):
    subexprs = ['operand']
    infix = True
    is_inc_dec_op = False

    def calculate_constant_result(self):
        func = compile_time_unary_operators[self.operator]
        self.constant_result = func(self.operand.constant_result)

    def compile_time_value(self, denv):
        func = compile_time_unary_operators.get(self.operator)
        if not func:
            error(self.pos, "Unary '%s' not supported in compile-time expression" % self.operator)
        operand = self.operand.compile_time_value(denv)
        try:
            return func(operand)
        except Exception as e:
            self.compile_time_value_error(e)

    def infer_type(self, env):
        operand_type = self.operand.infer_type(env)
        if operand_type.is_cpp_class or operand_type.is_ptr:
            cpp_type = operand_type.find_cpp_operation_type(self.operator)
            if cpp_type is not None:
                return cpp_type
        return self.infer_unop_type(env, operand_type)

    def infer_unop_type(self, env, operand_type):
        if operand_type.is_pyobject:
            return py_object_type
        else:
            return operand_type

    def may_be_none(self):
        if self.operand.type and self.operand.type.is_builtin_type:
            if self.operand.type is not type_type:
                return False
        return ExprNode.may_be_none(self)

    def analyse_types(self, env):
        self.operand = self.operand.analyse_types(env)
        if self.is_pythran_operation(env):
            self.type = PythranExpr(pythran_unaryop_type(self.operator, self.operand.type))
            self.is_temp = 1
        elif self.is_py_operation():
            self.coerce_operand_to_pyobject(env)
            self.type = py_object_type
            self.is_temp = 1
        elif self.is_cpp_operation():
            self.analyse_cpp_operation(env)
        else:
            self.analyse_c_operation(env)
        return self

    def check_const(self):
        return self.operand.check_const()

    def is_py_operation(self):
        return self.operand.type.is_pyobject or self.operand.type.is_ctuple

    def is_pythran_operation(self, env):
        np_pythran = has_np_pythran(env)
        op_type = self.operand.type
        return np_pythran and (op_type.is_buffer or op_type.is_pythran_expr)

    def nogil_check(self, env):
        if self.is_py_operation():
            self.gil_error()

    def is_cpp_operation(self):
        type = self.operand.type
        return type.is_cpp_class

    def coerce_operand_to_pyobject(self, env):
        self.operand = self.operand.coerce_to_pyobject(env)

    def generate_result_code(self, code):
        if self.type.is_pythran_expr:
            code.putln('// Pythran unaryop')
            code.putln('__Pyx_call_destructor(%s);' % self.result())
            code.putln('new (&%s) decltype(%s){%s%s};' % (self.result(), self.result(), self.operator, self.operand.pythran_result()))
        elif self.operand.type.is_pyobject:
            self.generate_py_operation_code(code)
        elif self.is_temp:
            if self.is_cpp_operation() and self.exception_check == '+':
                translate_cpp_exception(code, self.pos, '%s = %s %s;' % (self.result(), self.operator, self.operand.result()), self.result() if self.type.is_pyobject else None, self.exception_value, self.in_nogil_context)
            else:
                code.putln('%s = %s %s;' % (self.result(), self.operator, self.operand.result()))

    def generate_py_operation_code(self, code):
        function = self.py_operation_function(code)
        code.putln('%s = %s(%s); %s' % (self.result(), function, self.operand.py_result(), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)

    def type_error(self):
        if not self.operand.type.is_error:
            error(self.pos, "Invalid operand type for '%s' (%s)" % (self.operator, self.operand.type))
        self.type = PyrexTypes.error_type

    def analyse_cpp_operation(self, env, overload_check=True):
        operand_types = [self.operand.type]
        if self.is_inc_dec_op and (not self.is_prefix):
            operand_types.append(PyrexTypes.c_int_type)
        entry = env.lookup_operator_for_types(self.pos, self.operator, operand_types)
        if overload_check and (not entry):
            self.type_error()
            return
        if entry:
            self.exception_check = entry.type.exception_check
            self.exception_value = entry.type.exception_value
            if self.exception_check == '+':
                self.is_temp = True
                if needs_cpp_exception_conversion(self):
                    env.use_utility_code(UtilityCode.load_cached('CppExceptionConversion', 'CppSupport.cpp'))
        else:
            self.exception_check = ''
            self.exception_value = ''
        if self.is_inc_dec_op and (not self.is_prefix):
            cpp_type = self.operand.type.find_cpp_operation_type(self.operator, operand_type=PyrexTypes.c_int_type)
        else:
            cpp_type = self.operand.type.find_cpp_operation_type(self.operator)
        if overload_check and cpp_type is None:
            error(self.pos, "'%s' operator not defined for %s" % (self.operator, type))
            self.type_error()
            return
        self.type = cpp_type