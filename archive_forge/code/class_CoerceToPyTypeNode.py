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
class CoerceToPyTypeNode(CoercionNode):
    type = py_object_type
    target_type = py_object_type
    is_temp = 1

    def __init__(self, arg, env, type=py_object_type):
        if not arg.type.create_to_py_utility_code(env):
            error(arg.pos, "Cannot convert '%s' to Python object" % arg.type)
        elif arg.type.is_complex:
            arg = arg.coerce_to_simple(env)
        CoercionNode.__init__(self, arg)
        if type is py_object_type:
            if arg.type.is_string or arg.type.is_cpp_string:
                self.type = default_str_type(env)
            elif arg.type.is_pyunicode_ptr or arg.type.is_unicode_char:
                self.type = unicode_type
            elif arg.type.is_complex:
                self.type = Builtin.complex_type
            self.target_type = self.type
        elif arg.type.is_string or arg.type.is_cpp_string:
            if type not in (bytes_type, bytearray_type) and (not env.directives['c_string_encoding']):
                error(arg.pos, "default encoding required for conversion from '%s' to '%s'" % (arg.type, type))
            self.type = self.target_type = type
        else:
            self.target_type = type
    gil_message = 'Converting to Python object'

    def may_be_none(self):
        return False

    def coerce_to_boolean(self, env):
        arg_type = self.arg.type
        if arg_type == PyrexTypes.c_bint_type or (arg_type.is_pyobject and arg_type.name == 'bool'):
            return self.arg.coerce_to_temp(env)
        else:
            return CoerceToBooleanNode(self, env)

    def coerce_to_integer(self, env):
        if self.arg.type.is_int:
            return self.arg
        else:
            return self.arg.coerce_to(PyrexTypes.c_long_type, env)

    def analyse_types(self, env):
        return self

    def generate_result_code(self, code):
        code.putln('%s; %s' % (self.arg.type.to_py_call_code(self.arg.result(), self.result(), self.target_type), code.error_goto_if_null(self.result(), self.pos)))
        self.generate_gotref(code)