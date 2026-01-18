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
class CTypeDefNode(StatNode):
    child_attrs = ['base_type', 'declarator']

    def analyse_declarations(self, env):
        base = self.base_type.analyse(env)
        name_declarator, type = self.declarator.analyse(base, env, visibility=self.visibility, in_pxd=self.in_pxd)
        name = name_declarator.name
        cname = name_declarator.cname
        entry = env.declare_typedef(name, type, self.pos, cname=cname, visibility=self.visibility, api=self.api)
        if type.is_fused:
            entry.in_cinclude = True
        if self.in_pxd and (not env.in_cinclude):
            entry.defined_in_pxd = 1

    def analyse_expressions(self, env):
        return self

    def generate_execution_code(self, code):
        pass