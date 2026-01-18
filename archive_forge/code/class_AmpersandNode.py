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
class AmpersandNode(CUnopNode):
    operator = '&'

    def infer_unop_type(self, env, operand_type):
        return PyrexTypes.c_ptr_type(operand_type)

    def analyse_types(self, env):
        self.operand = self.operand.analyse_types(env)
        argtype = self.operand.type
        if argtype.is_cpp_class:
            self.analyse_cpp_operation(env, overload_check=False)
        if not (argtype.is_cfunction or argtype.is_reference or self.operand.is_addressable()):
            if argtype.is_memoryviewslice:
                self.error('Cannot take address of memoryview slice')
            else:
                self.error('Taking address of non-lvalue (type %s)' % argtype)
            return self
        if argtype.is_pyobject:
            self.error('Cannot take address of Python %s' % ("variable '%s'" % self.operand.name if self.operand.is_name else "object attribute '%s'" % self.operand.attribute if self.operand.is_attribute else 'object'))
            return self
        if not argtype.is_cpp_class or not self.type:
            self.type = PyrexTypes.c_ptr_type(argtype)
        return self

    def check_const(self):
        return self.operand.check_const_addr()

    def error(self, mess):
        error(self.pos, mess)
        self.type = PyrexTypes.error_type
        self.result_code = '<error>'

    def calculate_result_code(self):
        return '(&%s)' % self.operand.result()

    def generate_result_code(self, code):
        if self.operand.type.is_cpp_class and self.exception_check == '+':
            translate_cpp_exception(code, self.pos, '%s = %s %s;' % (self.result(), self.operator, self.operand.result()), self.result() if self.type.is_pyobject else None, self.exception_value, self.in_nogil_context)