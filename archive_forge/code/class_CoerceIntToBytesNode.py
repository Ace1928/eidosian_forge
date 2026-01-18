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
class CoerceIntToBytesNode(CoerceToPyTypeNode):
    is_temp = 1

    def __init__(self, arg, env):
        arg = arg.coerce_to_simple(env)
        CoercionNode.__init__(self, arg)
        self.type = Builtin.bytes_type

    def generate_result_code(self, code):
        arg = self.arg
        arg_result = arg.result()
        if arg.type not in (PyrexTypes.c_char_type, PyrexTypes.c_uchar_type, PyrexTypes.c_schar_type):
            if arg.type.signed:
                code.putln('if ((%s < 0) || (%s > 255)) {' % (arg_result, arg_result))
            else:
                code.putln('if (%s > 255) {' % arg_result)
            code.putln('PyErr_SetString(PyExc_OverflowError, "value too large to pack into a byte"); %s' % code.error_goto(self.pos))
            code.putln('}')
        temp = None
        if arg.type is not PyrexTypes.c_char_type:
            temp = code.funcstate.allocate_temp(PyrexTypes.c_char_type, manage_ref=False)
            code.putln('%s = (char)%s;' % (temp, arg_result))
            arg_result = temp
        code.putln('%s = PyBytes_FromStringAndSize(&%s, 1); %s' % (self.result(), arg_result, code.error_goto_if_null(self.result(), self.pos)))
        if temp is not None:
            code.funcstate.release_temp(temp)
        self.generate_gotref(code)