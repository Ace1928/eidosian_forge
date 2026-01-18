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
class SwitchStatNode(StatNode):
    child_attrs = ['test', 'cases', 'else_clause']

    def generate_execution_code(self, code):
        self.test.generate_evaluation_code(code)
        for case in self.cases:
            case.generate_condition_evaluation_code(code)
        code.mark_pos(self.pos)
        code.putln('switch (%s) {' % self.test.result())
        for case in self.cases:
            case.generate_execution_code(code)
        if self.else_clause is not None:
            code.putln('default:')
            self.else_clause.generate_execution_code(code)
            code.putln('break;')
        else:
            code.putln('default: break;')
        code.putln('}')
        self.test.generate_disposal_code(code)
        self.test.free_temps(code)

    def generate_function_definitions(self, env, code):
        self.test.generate_function_definitions(env, code)
        for case in self.cases:
            case.generate_function_definitions(env, code)
        if self.else_clause is not None:
            self.else_clause.generate_function_definitions(env, code)

    def annotate(self, code):
        self.test.annotate(code)
        for case in self.cases:
            case.annotate(code)
        if self.else_clause is not None:
            self.else_clause.annotate(code)