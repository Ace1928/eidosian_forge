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
class CoerceToBooleanNode(CoercionNode):
    type = PyrexTypes.c_bint_type
    _special_builtins = {Builtin.list_type: 'PyList_GET_SIZE', Builtin.tuple_type: 'PyTuple_GET_SIZE', Builtin.set_type: 'PySet_GET_SIZE', Builtin.frozenset_type: 'PySet_GET_SIZE', Builtin.bytes_type: 'PyBytes_GET_SIZE', Builtin.bytearray_type: 'PyByteArray_GET_SIZE', Builtin.unicode_type: '__Pyx_PyUnicode_IS_TRUE'}

    def __init__(self, arg, env):
        CoercionNode.__init__(self, arg)
        if arg.type.is_pyobject:
            self.is_temp = 1

    def nogil_check(self, env):
        if self.arg.type.is_pyobject and self._special_builtins.get(self.arg.type) is None:
            self.gil_error()
    gil_message = 'Truth-testing Python object'

    def check_const(self):
        if self.is_temp:
            self.not_const()
            return False
        return self.arg.check_const()

    def calculate_result_code(self):
        return '(%s != 0)' % self.arg.result()

    def generate_result_code(self, code):
        if not self.is_temp:
            return
        test_func = self._special_builtins.get(self.arg.type)
        if test_func is not None:
            checks = ['(%s != Py_None)' % self.arg.py_result()] if self.arg.may_be_none() else []
            checks.append('(%s(%s) != 0)' % (test_func, self.arg.py_result()))
            code.putln('%s = %s;' % (self.result(), '&&'.join(checks)))
        else:
            code.putln('%s = __Pyx_PyObject_IsTrue(%s); %s' % (self.result(), self.arg.py_result(), code.error_goto_if_neg(self.result(), self.pos)))

    def analyse_types(self, env):
        return self