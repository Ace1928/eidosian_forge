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
class SwitchCaseNode(StatNode):
    child_attrs = ['conditions', 'body']

    def generate_condition_evaluation_code(self, code):
        for cond in self.conditions:
            cond.generate_evaluation_code(code)

    def generate_execution_code(self, code):
        num_conditions = len(self.conditions)
        line_tracing_enabled = code.globalstate.directives['linetrace']
        for i, cond in enumerate(self.conditions, 1):
            code.putln('case %s:' % cond.result())
            code.mark_pos(cond.pos)
            if line_tracing_enabled and i < num_conditions:
                code.putln('CYTHON_FALLTHROUGH;')
        self.body.generate_execution_code(code)
        code.mark_pos(self.pos, trace=False)
        code.putln('break;')

    def generate_function_definitions(self, env, code):
        for cond in self.conditions:
            cond.generate_function_definitions(env, code)
        self.body.generate_function_definitions(env, code)

    def annotate(self, code):
        for cond in self.conditions:
            cond.annotate(code)
        self.body.annotate(code)