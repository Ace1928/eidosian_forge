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
class WithTargetAssignmentStatNode(AssignmentNode):
    child_attrs = ['rhs', 'lhs']
    with_node = None
    rhs = None

    def analyse_declarations(self, env):
        self.lhs.analyse_target_declaration(env)

    def analyse_expressions(self, env):
        self.lhs = self.lhs.analyse_target_types(env)
        self.lhs.gil_assignment_check(env)
        self.rhs = self.with_node.target_temp.coerce_to(self.lhs.type, env)
        return self

    def generate_execution_code(self, code):
        self.rhs.generate_evaluation_code(code)
        self.lhs.generate_assignment_code(self.rhs, code)
        self.with_node.target_temp.release(code)

    def annotate(self, code):
        self.lhs.annotate(code)
        self.rhs.annotate(code)