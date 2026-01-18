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
def _allocate_closure_temp(self, code, entry):
    """
        Helper function that allocate a temporary for a closure variable that
        is assigned to.
        """
    if self.parent:
        return self.parent._allocate_closure_temp(code, entry)
    if entry.cname in self.seen_closure_vars:
        return entry.cname
    cname = code.funcstate.allocate_temp(entry.type, True)
    self.seen_closure_vars.add(entry.cname)
    self.seen_closure_vars.add(cname)
    self.modified_entries.append((entry, entry.cname))
    code.putln('%s = %s;' % (cname, entry.cname))
    entry.cname = cname