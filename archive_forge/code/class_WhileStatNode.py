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
class WhileStatNode(LoopNode, StatNode):
    child_attrs = ['condition', 'body', 'else_clause']

    def analyse_declarations(self, env):
        self.body.analyse_declarations(env)
        if self.else_clause:
            self.else_clause.analyse_declarations(env)

    def analyse_expressions(self, env):
        if self.condition:
            self.condition = self.condition.analyse_temp_boolean_expression(env)
        self.body = self.body.analyse_expressions(env)
        if self.else_clause:
            self.else_clause = self.else_clause.analyse_expressions(env)
        return self

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        old_loop_labels = code.new_loop_labels()
        code.putln('while (1) {')
        if self.condition:
            self.condition.generate_evaluation_code(code)
            self.condition.generate_disposal_code(code)
            code.putln('if (!%s) break;' % self.condition.result())
            self.condition.free_temps(code)
        self.body.generate_execution_code(code)
        code.put_label(code.continue_label)
        code.putln('}')
        break_label = code.break_label
        code.set_loop_labels(old_loop_labels)
        if self.else_clause:
            code.mark_pos(self.else_clause.pos)
            code.putln('/*else*/ {')
            self.else_clause.generate_execution_code(code)
            code.putln('}')
        code.put_label(break_label)

    def generate_function_definitions(self, env, code):
        if self.condition:
            self.condition.generate_function_definitions(env, code)
        self.body.generate_function_definitions(env, code)
        if self.else_clause is not None:
            self.else_clause.generate_function_definitions(env, code)

    def annotate(self, code):
        if self.condition:
            self.condition.annotate(code)
        self.body.annotate(code)
        if self.else_clause:
            self.else_clause.annotate(code)