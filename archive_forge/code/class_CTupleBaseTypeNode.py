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
class CTupleBaseTypeNode(CBaseTypeNode):
    child_attrs = ['components']

    def analyse(self, env, could_be_name=False):
        component_types = []
        for c in self.components:
            type = c.analyse(env)
            if type.is_pyobject:
                error(c.pos, "Tuple types can't (yet) contain Python objects.")
                return error_type
            component_types.append(type)
        entry = env.declare_tuple_type(self.pos, component_types)
        entry.used = True
        return entry.type