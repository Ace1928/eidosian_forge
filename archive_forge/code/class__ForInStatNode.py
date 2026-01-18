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
class _ForInStatNode(LoopNode, StatNode):
    child_attrs = ['target', 'item', 'iterator', 'body', 'else_clause']
    item = None
    is_async = False

    def _create_item_node(self):
        raise NotImplementedError('must be implemented by subclasses')

    def analyse_declarations(self, env):
        self.target.analyse_target_declaration(env)
        self.body.analyse_declarations(env)
        if self.else_clause:
            self.else_clause.analyse_declarations(env)
        self._create_item_node()

    def analyse_expressions(self, env):
        self.target = self.target.analyse_target_types(env)
        self.iterator = self.iterator.analyse_expressions(env)
        self._create_item_node()
        self.item = self.item.analyse_expressions(env)
        if not self.is_async and (self.iterator.type.is_ptr or self.iterator.type.is_array) and self.target.type.assignable_from(self.iterator.type):
            pass
        else:
            self.item = self.item.coerce_to(self.target.type, env)
        self.body = self.body.analyse_expressions(env)
        if self.else_clause:
            self.else_clause = self.else_clause.analyse_expressions(env)
        return self

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        old_loop_labels = code.new_loop_labels()
        self.iterator.generate_evaluation_code(code)
        code.putln('for (;;) {')
        self.item.generate_evaluation_code(code)
        self.target.generate_assignment_code(self.item, code)
        self.body.generate_execution_code(code)
        code.mark_pos(self.pos)
        code.put_label(code.continue_label)
        code.putln('}')
        self.iterator.generate_disposal_code(code)
        else_label = code.new_label('for_else') if self.else_clause else None
        end_label = code.new_label('for_end')
        label_intercepts = code.label_interceptor([code.break_label], [end_label], skip_to_label=else_label or end_label, pos=self.pos)
        code.mark_pos(self.pos)
        for _ in label_intercepts:
            self.iterator.generate_disposal_code(code)
        code.set_loop_labels(old_loop_labels)
        self.iterator.free_temps(code)
        if self.else_clause:
            code.putln('/*else*/ {')
            code.put_label(else_label)
            self.else_clause.generate_execution_code(code)
            code.putln('}')
        code.put_label(end_label)

    def generate_function_definitions(self, env, code):
        self.target.generate_function_definitions(env, code)
        self.iterator.generate_function_definitions(env, code)
        self.body.generate_function_definitions(env, code)
        if self.else_clause is not None:
            self.else_clause.generate_function_definitions(env, code)

    def annotate(self, code):
        self.target.annotate(code)
        self.iterator.annotate(code)
        self.body.annotate(code)
        if self.else_clause:
            self.else_clause.annotate(code)
        self.item.annotate(code)