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
def _generate_break(self, code):
    code.globalstate.use_utility_code(UtilityCode.load_cached('StopAsyncIteration', 'Coroutine.c'))
    code.putln('PyObject* exc_type = __Pyx_PyErr_CurrentExceptionType();')
    code.putln('if (unlikely(exc_type && (exc_type == __Pyx_PyExc_StopAsyncIteration || ( exc_type != PyExc_StopIteration && exc_type != PyExc_GeneratorExit && __Pyx_PyErr_GivenExceptionMatches(exc_type, __Pyx_PyExc_StopAsyncIteration))))) {')
    code.putln('PyErr_Clear();')
    code.putln('break;')
    code.putln('}')