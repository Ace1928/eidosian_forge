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
class WithStatNode(StatNode):
    """
    Represents a Python with statement.

    Implemented by the WithTransform as follows:

        MGR = EXPR
        EXIT = MGR.__exit__
        VALUE = MGR.__enter__()
        EXC = True
        try:
            try:
                TARGET = VALUE  # optional
                BODY
            except:
                EXC = False
                if not EXIT(*EXCINFO):
                    raise
        finally:
            if EXC:
                EXIT(None, None, None)
            MGR = EXIT = VALUE = None
    """
    child_attrs = ['manager', 'enter_call', 'target', 'body']
    enter_call = None
    target_temp = None

    def analyse_declarations(self, env):
        self.manager.analyse_declarations(env)
        self.enter_call.analyse_declarations(env)
        self.body.analyse_declarations(env)

    def analyse_expressions(self, env):
        self.manager = self.manager.analyse_types(env)
        self.enter_call = self.enter_call.analyse_types(env)
        if self.target:
            from .ExprNodes import TempNode
            self.target_temp = TempNode(self.enter_call.pos, self.enter_call.type)
        self.body = self.body.analyse_expressions(env)
        return self

    def generate_function_definitions(self, env, code):
        self.manager.generate_function_definitions(env, code)
        self.enter_call.generate_function_definitions(env, code)
        self.body.generate_function_definitions(env, code)

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        code.putln('/*with:*/ {')
        self.manager.generate_evaluation_code(code)
        self.exit_var = code.funcstate.allocate_temp(py_object_type, manage_ref=False)
        code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectLookupSpecial', 'ObjectHandling.c'))
        code.putln('%s = __Pyx_PyObject_LookupSpecial(%s, %s); %s' % (self.exit_var, self.manager.py_result(), code.intern_identifier(EncodedString('__aexit__' if self.is_async else '__exit__')), code.error_goto_if_null(self.exit_var, self.pos)))
        code.put_gotref(self.exit_var, py_object_type)
        old_error_label = code.new_error_label()
        intermediate_error_label = code.error_label
        self.enter_call.generate_evaluation_code(code)
        if self.target:
            self.target_temp.allocate(code)
            self.enter_call.make_owned_reference(code)
            code.putln('%s = %s;' % (self.target_temp.result(), self.enter_call.result()))
            self.enter_call.generate_post_assignment_code(code)
        else:
            self.enter_call.generate_disposal_code(code)
        self.enter_call.free_temps(code)
        self.manager.generate_disposal_code(code)
        self.manager.free_temps(code)
        code.error_label = old_error_label
        self.body.generate_execution_code(code)
        if code.label_used(intermediate_error_label):
            step_over_label = code.new_label()
            code.put_goto(step_over_label)
            code.put_label(intermediate_error_label)
            code.put_decref_clear(self.exit_var, py_object_type)
            code.put_goto(old_error_label)
            code.put_label(step_over_label)
        code.funcstate.release_temp(self.exit_var)
        code.putln('}')