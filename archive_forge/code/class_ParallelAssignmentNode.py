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
class ParallelAssignmentNode(AssignmentNode):
    child_attrs = ['stats']

    def analyse_declarations(self, env):
        for stat in self.stats:
            stat.analyse_declarations(env)

    def analyse_expressions(self, env):
        self.stats = [stat.analyse_types(env, use_temp=1) for stat in self.stats]
        for stat in self.stats:
            stat._check_const_assignment(stat)
        return self

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        for stat in self.stats:
            stat.generate_rhs_evaluation_code(code)
        for stat in self.stats:
            stat.generate_assignment_code(code)

    def generate_function_definitions(self, env, code):
        for stat in self.stats:
            stat.generate_function_definitions(env, code)

    def annotate(self, code):
        for stat in self.stats:
            stat.annotate(code)