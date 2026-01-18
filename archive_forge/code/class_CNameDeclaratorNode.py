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
class CNameDeclaratorNode(CDeclaratorNode):
    child_attrs = ['default']
    default = None

    def declared_name(self):
        return self.name

    def analyse(self, base_type, env, nonempty=0, visibility=None, in_pxd=False):
        if nonempty and self.name == '':
            if base_type.is_ptr or base_type.is_array or base_type.is_buffer:
                error(self.pos, 'Missing argument name')
            elif base_type.is_void:
                error(self.pos, 'Use spam() rather than spam(void) to declare a function with no arguments.')
            else:
                self.name = base_type.declaration_code('', for_display=1, pyrex=1)
                base_type = py_object_type
        if base_type.is_fused and env.fused_to_specific:
            try:
                base_type = base_type.specialize(env.fused_to_specific)
            except CannotSpecialize:
                error(self.pos, "'%s' cannot be specialized since its type is not a fused argument to this function" % self.name)
        self.type = base_type
        return (self, base_type)