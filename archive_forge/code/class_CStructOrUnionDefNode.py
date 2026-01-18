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
class CStructOrUnionDefNode(StatNode):
    child_attrs = ['attributes']

    def declare(self, env, scope=None):
        self.entry = env.declare_struct_or_union(self.name, self.kind, scope, self.typedef_flag, self.pos, self.cname, visibility=self.visibility, api=self.api, packed=self.packed)

    def analyse_declarations(self, env):
        scope = None
        if self.attributes is not None:
            scope = StructOrUnionScope(self.name)
        self.declare(env, scope)
        if self.attributes is not None:
            if self.in_pxd and (not env.in_cinclude):
                self.entry.defined_in_pxd = 1
            for attr in self.attributes:
                attr.analyse_declarations(env, scope)
            if self.visibility != 'extern':
                for attr in scope.var_entries:
                    type = attr.type
                    while type.is_array:
                        type = type.base_type
                    if type == self.entry.type:
                        error(attr.pos, 'Struct cannot contain itself as a member.')

    def analyse_expressions(self, env):
        return self

    def generate_execution_code(self, code):
        pass