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
class CoerceFromPyTypeNode(CoercionNode):
    special_none_cvalue = None

    def __init__(self, result_type, arg, env):
        CoercionNode.__init__(self, arg)
        self.type = result_type
        self.is_temp = 1
        if not result_type.create_from_py_utility_code(env):
            error(arg.pos, "Cannot convert Python object to '%s'" % result_type)
        if self.type.is_string or self.type.is_pyunicode_ptr:
            if self.arg.is_name and self.arg.entry and self.arg.entry.is_pyglobal:
                warning(arg.pos, "Obtaining '%s' from externally modifiable global Python value" % result_type, level=1)
            if self.type.is_pyunicode_ptr:
                warning(arg.pos, 'Py_UNICODE* has been removed in Python 3.12. This conversion to a Py_UNICODE* will no longer compile in the latest Python versions. Use Python C API functions like PyUnicode_AsWideCharString if you need to obtain a wchar_t* on Windows (and free the string manually after use).', level=1)

    def analyse_types(self, env):
        return self

    def is_ephemeral(self):
        return (self.type.is_ptr and (not self.type.is_array)) and self.arg.is_ephemeral()

    def generate_result_code(self, code):
        from_py_function = None
        if self.type.is_string and self.arg.type is bytes_type:
            if self.type.from_py_function.startswith('__Pyx_PyObject_As'):
                from_py_function = '__Pyx_PyBytes' + self.type.from_py_function[len('__Pyx_PyObject'):]
                NoneCheckNode.generate_if_needed(self.arg, code, 'expected bytes, NoneType found')
        code.putln(self.type.from_py_call_code(self.arg.py_result(), self.result(), self.pos, code, from_py_function=from_py_function, special_none_cvalue=self.special_none_cvalue))
        if self.type.is_pyobject:
            self.generate_gotref(code)

    def nogil_check(self, env):
        error(self.pos, 'Coercion from Python not allowed without the GIL')