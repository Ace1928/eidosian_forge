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
class CPropertyNode(StatNode):
    """Definition of a C property, backed by a CFuncDefNode getter.
    """
    child_attrs = ['body']
    is_cproperty = True

    @property
    def cfunc(self):
        stats = self.body.stats
        assert stats and isinstance(stats[0], CFuncDefNode), stats
        return stats[0]

    def analyse_declarations(self, env):
        scope = PropertyScope(self.name, class_scope=env)
        self.body.analyse_declarations(scope)
        entry = self.entry = env.declare_property(self.name, self.doc, self.pos, ctype=self.cfunc.return_type, property_scope=scope)
        entry.getter_cname = self.cfunc.entry.cname

    def analyse_expressions(self, env):
        self.body = self.body.analyse_expressions(env)
        return self

    def generate_function_definitions(self, env, code):
        self.body.generate_function_definitions(env, code)

    def generate_execution_code(self, code):
        pass

    def annotate(self, code):
        self.body.annotate(code)