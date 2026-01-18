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
def _check_byte_value(self, code, rhs):
    assert rhs.type.is_int, repr(rhs.type)
    value_code = rhs.result()
    if rhs.has_constant_result():
        if 0 <= rhs.constant_result < 256:
            return value_code
        needs_cast = True
        warning(rhs.pos, 'value outside of range(0, 256) when assigning to byte: %s' % rhs.constant_result, level=1)
    else:
        needs_cast = rhs.type != PyrexTypes.c_uchar_type
    if not self.nogil:
        conditions = []
        if rhs.is_literal or rhs.type.signed:
            conditions.append('%s < 0' % value_code)
        if rhs.is_literal or not (rhs.is_temp and rhs.type in (PyrexTypes.c_uchar_type, PyrexTypes.c_char_type, PyrexTypes.c_schar_type)):
            conditions.append('%s > 255' % value_code)
        if conditions:
            code.putln('if (unlikely(%s)) {' % ' || '.join(conditions))
            code.putln('PyErr_SetString(PyExc_ValueError, "byte must be in range(0, 256)"); %s' % code.error_goto(self.pos))
            code.putln('}')
    if needs_cast:
        value_code = '((unsigned char)%s)' % value_code
    return value_code