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
class TypeidNode(ExprNode):
    subexprs = ['operand']
    arg_type = None
    is_variable = None
    is_temp = 1

    def get_type_info_type(self, env):
        env_module = env
        while not env_module.is_module_scope:
            env_module = env_module.outer_scope
        typeinfo_module = env_module.find_module('libcpp.typeinfo', self.pos)
        typeinfo_entry = typeinfo_module.lookup('type_info')
        return PyrexTypes.CFakeReferenceType(PyrexTypes.c_const_type(typeinfo_entry.type))
    cpp_message = 'typeid operator'

    def analyse_types(self, env):
        if not self.type:
            self.type = PyrexTypes.error_type
        self.cpp_check(env)
        type_info = self.get_type_info_type(env)
        if not type_info:
            self.error("The 'libcpp.typeinfo' module must be cimported to use the typeid() operator")
            return self
        if self.operand is None:
            return self
        self.type = type_info
        as_type = self.operand.analyse_as_specialized_type(env)
        if as_type:
            self.arg_type = as_type
            self.is_type = True
            self.operand = None
        else:
            self.arg_type = self.operand.analyse_types(env)
            self.is_type = False
            self.operand = None
            if self.arg_type.type.is_pyobject:
                self.error('Cannot use typeid on a Python object')
                return self
            elif self.arg_type.type.is_void:
                self.error('Cannot use typeid on void')
                return self
            elif not self.arg_type.type.is_complete():
                self.error("Cannot use typeid on incomplete type '%s'" % self.arg_type.type)
                return self
        env.use_utility_code(UtilityCode.load_cached('CppExceptionConversion', 'CppSupport.cpp'))
        return self

    def error(self, mess):
        error(self.pos, mess)
        self.type = PyrexTypes.error_type
        self.result_code = '<error>'

    def check_const(self):
        return True

    def calculate_result_code(self):
        return self.temp_code

    def generate_result_code(self, code):
        if self.is_type:
            arg_code = self.arg_type.empty_declaration_code()
        else:
            arg_code = self.arg_type.result()
        translate_cpp_exception(code, self.pos, '%s = typeid(%s);' % (self.temp_code, arg_code), None, None, self.in_nogil_context)