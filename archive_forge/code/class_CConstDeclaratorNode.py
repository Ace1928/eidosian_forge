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
class CConstDeclaratorNode(CDeclaratorNode):
    child_attrs = ['base']

    def analyse(self, base_type, env, nonempty=0, visibility=None, in_pxd=False):
        if base_type.is_pyobject:
            error(self.pos, 'Const base type cannot be a Python object')
        const = PyrexTypes.c_const_type(base_type)
        return self.base.analyse(const, env, nonempty=nonempty, visibility=visibility, in_pxd=in_pxd)