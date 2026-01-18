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
class IfStatNode(StatNode):
    child_attrs = ['if_clauses', 'else_clause']

    def analyse_declarations(self, env):
        for if_clause in self.if_clauses:
            if_clause.analyse_declarations(env)
        if self.else_clause:
            self.else_clause.analyse_declarations(env)

    def analyse_expressions(self, env):
        self.if_clauses = [if_clause.analyse_expressions(env) for if_clause in self.if_clauses]
        if self.else_clause:
            self.else_clause = self.else_clause.analyse_expressions(env)
        return self

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        end_label = code.new_label()
        last = len(self.if_clauses)
        if not self.else_clause:
            last -= 1
        for i, if_clause in enumerate(self.if_clauses):
            if_clause.generate_execution_code(code, end_label, is_last=i == last)
        if self.else_clause:
            code.mark_pos(self.else_clause.pos)
            code.putln('/*else*/ {')
            self.else_clause.generate_execution_code(code)
            code.putln('}')
        code.put_label(end_label)

    def generate_function_definitions(self, env, code):
        for clause in self.if_clauses:
            clause.generate_function_definitions(env, code)
        if self.else_clause is not None:
            self.else_clause.generate_function_definitions(env, code)

    def annotate(self, code):
        for if_clause in self.if_clauses:
            if_clause.annotate(code)
        if self.else_clause:
            self.else_clause.annotate(code)