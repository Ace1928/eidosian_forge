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
class PreImportScope(Scope):
    namespace_cname = Naming.preimport_cname

    def __init__(self):
        Scope.__init__(self, Options.pre_import, None, None)

    def declare_builtin(self, name, pos):
        entry = self.declare(name, name, py_object_type, pos, 'private')
        entry.is_variable = True
        entry.is_pyglobal = True
        return entry