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
class CNestedBaseTypeNode(CBaseTypeNode):
    child_attrs = ['base_type']

    def analyse(self, env, could_be_name=None):
        base_type = self.base_type.analyse(env)
        if base_type is PyrexTypes.error_type:
            return PyrexTypes.error_type
        if not base_type.is_cpp_class:
            error(self.pos, "'%s' is not a valid type scope" % base_type)
            return PyrexTypes.error_type
        type_entry = base_type.scope.lookup_here(self.name)
        if not type_entry or not type_entry.is_type:
            error(self.pos, "'%s.%s' is not a type identifier" % (base_type, self.name))
            return PyrexTypes.error_type
        return type_entry.type