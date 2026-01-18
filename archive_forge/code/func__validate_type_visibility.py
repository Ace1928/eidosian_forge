from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
def _validate_type_visibility(self, type, pos, env):
    """
        Ensure that types used in cdef functions are public or api, or
        defined in a C header.
        """
    public_or_api = self.visibility == 'public' or self.api
    entry = getattr(type, 'entry', None)
    if public_or_api and entry and env.is_module_scope:
        if not (entry.visibility in ('public', 'extern') or entry.api or entry.in_cinclude):
            error(pos, 'Function declared public or api may not have private types')