from __future__ import absolute_import
import re
import copy
import operator
from ..Utils import try_finally_contextmanager
from .Errors import warning, error, InternalError, performance_hint
from .StringEncoding import EncodedString
from . import Options, Naming
from . import PyrexTypes
from .PyrexTypes import py_object_type, unspecified_type
from .TypeSlots import (
from . import Future
from . import Code
def _create_inner_entry_for_closure(self, name, entry):
    entry.in_closure = True
    inner_entry = InnerEntry(entry, self)
    inner_entry.is_variable = True
    self.entries[name] = inner_entry
    return inner_entry