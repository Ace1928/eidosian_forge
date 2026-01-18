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
class TypecastNode(ExprNode):
    subexprs = ['operand']
    base_type = declarator = type = None

    def type_dependencies(self, env):
        return ()

    def infer_type(self, env):
        if self.type is None:
            base_type = self.base_type.analyse(env)
            _, self.type = self.declarator.analyse(base_type, env)
        return self.type

    def analyse_types(self, env):
        if self.type is None:
            base_type = self.base_type.analyse(env)
            _, self.type = self.declarator.analyse(base_type, env)
        if self.operand.has_constant_result():
            self.calculate_constant_result()
        if self.type.is_cfunction:
            error(self.pos, 'Cannot cast to a function type')
            self.type = PyrexTypes.error_type
        self.operand = self.operand.analyse_types(env)
        if self.type is PyrexTypes.c_bint_type:
            return self.operand.coerce_to_boolean(env)
        to_py = self.type.is_pyobject
        from_py = self.operand.type.is_pyobject
        if from_py and (not to_py) and self.operand.is_ephemeral():
            if not self.type.is_numeric and (not self.type.is_cpp_class):
                error(self.pos, 'Casting temporary Python object to non-numeric non-Python type')
        if to_py and (not from_py):
            if self.type is bytes_type and self.operand.type.is_int:
                return CoerceIntToBytesNode(self.operand, env)
            elif self.operand.type.can_coerce_to_pyobject(env):
                self.result_ctype = py_object_type
                self.operand = self.operand.coerce_to(self.type, env)
            else:
                if self.operand.type.is_ptr:
                    if not (self.operand.type.base_type.is_void or self.operand.type.base_type.is_struct):
                        error(self.pos, 'Python objects cannot be cast from pointers of primitive types')
                else:
                    warning(self.pos, 'No conversion from %s to %s, python object pointer used.' % (self.operand.type, self.type))
                self.operand = self.operand.coerce_to_simple(env)
        elif from_py and (not to_py):
            if self.type.create_from_py_utility_code(env):
                self.operand = self.operand.coerce_to(self.type, env)
            elif self.type.is_ptr:
                if not (self.type.base_type.is_void or self.type.base_type.is_struct):
                    error(self.pos, 'Python objects cannot be cast to pointers of primitive types')
            else:
                warning(self.pos, 'No conversion from %s to %s, python object pointer used.' % (self.type, self.operand.type))
        elif from_py and to_py:
            if self.typecheck:
                self.operand = PyTypeTestNode(self.operand, self.type, env, notnone=True)
            elif isinstance(self.operand, SliceIndexNode):
                self.operand = self.operand.coerce_to(self.type, env)
        elif self.type.is_complex and self.operand.type.is_complex:
            self.operand = self.operand.coerce_to_simple(env)
        elif self.operand.type.is_fused:
            self.operand = self.operand.coerce_to(self.type, env)
        if self.type.is_ptr and self.type.base_type.is_cfunction and self.type.base_type.nogil:
            op_type = self.operand.type
            if op_type.is_ptr:
                op_type = op_type.base_type
            if op_type.is_cfunction and (not op_type.nogil):
                warning(self.pos, 'Casting a GIL-requiring function into a nogil function circumvents GIL validation', 1)
        return self

    def is_simple(self):
        return self.operand.is_simple()

    def is_ephemeral(self):
        return self.operand.is_ephemeral()

    def nonlocally_immutable(self):
        return self.is_temp or self.operand.nonlocally_immutable()

    def nogil_check(self, env):
        if self.type and self.type.is_pyobject and self.is_temp:
            self.gil_error()

    def check_const(self):
        return self.operand.check_const()

    def calculate_constant_result(self):
        self.constant_result = self.calculate_result_code(self.operand.constant_result)

    def calculate_result_code(self, operand_result=None):
        if operand_result is None:
            operand_result = self.operand.result()
        if self.type.is_complex:
            operand_result = self.operand.result()
            if self.operand.type.is_complex:
                real_part = self.type.real_type.cast_code(self.operand.type.real_code(operand_result))
                imag_part = self.type.real_type.cast_code(self.operand.type.imag_code(operand_result))
            else:
                real_part = self.type.real_type.cast_code(operand_result)
                imag_part = '0'
            return '%s(%s, %s)' % (self.type.from_parts, real_part, imag_part)
        else:
            return self.type.cast_code(operand_result)

    def get_constant_c_result_code(self):
        operand_result = self.operand.get_constant_c_result_code()
        if operand_result:
            return self.type.cast_code(operand_result)

    def result_as(self, type):
        if self.type.is_pyobject and (not self.is_temp):
            return self.operand.result_as(type)
        else:
            return ExprNode.result_as(self, type)

    def generate_result_code(self, code):
        if self.is_temp:
            code.putln('%s = (PyObject *)%s;' % (self.result(), self.operand.result()))
            code.put_incref(self.result(), self.ctype())