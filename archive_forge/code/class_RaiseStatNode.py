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
class RaiseStatNode(StatNode):
    child_attrs = ['exc_type', 'exc_value', 'exc_tb', 'cause']
    is_terminator = True
    builtin_exc_name = None
    wrap_tuple_value = False
    in_try_block = False

    def analyse_expressions(self, env):
        if self.exc_type:
            exc_type = self.exc_type.analyse_types(env)
            self.exc_type = exc_type.coerce_to_pyobject(env)
        if self.exc_value:
            exc_value = self.exc_value.analyse_types(env)
            if self.wrap_tuple_value:
                if exc_value.type is Builtin.tuple_type or not exc_value.type.is_builtin_type:
                    from .ExprNodes import TupleNode
                    exc_value = TupleNode(exc_value.pos, args=[exc_value.coerce_to_pyobject(env)], slow=True)
                    exc_value = exc_value.analyse_types(env, skip_children=True)
            self.exc_value = exc_value.coerce_to_pyobject(env)
        if self.exc_tb:
            exc_tb = self.exc_tb.analyse_types(env)
            self.exc_tb = exc_tb.coerce_to_pyobject(env)
        if self.cause:
            cause = self.cause.analyse_types(env)
            self.cause = cause.coerce_to_pyobject(env)
        if self.exc_type and (not self.exc_value) and (not self.exc_tb):
            exc = self.exc_type
            from . import ExprNodes
            if isinstance(exc, ExprNodes.SimpleCallNode) and (not (exc.args or (exc.arg_tuple is not None and exc.arg_tuple.args))):
                exc = exc.function
            if exc.is_name and exc.entry.is_builtin:
                from . import Symtab
                self.builtin_exc_name = exc.name
                if self.builtin_exc_name == 'MemoryError':
                    self.exc_type = None
                elif self.builtin_exc_name == 'StopIteration' and env.is_local_scope and (env.name == '__next__') and env.parent_scope and env.parent_scope.is_c_class_scope and (not self.in_try_block):
                    self.exc_type = None
        return self
    nogil_check = Node.gil_error
    gil_message = 'Raising exception'

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        if self.builtin_exc_name == 'MemoryError':
            code.putln('PyErr_NoMemory(); %s' % code.error_goto(self.pos))
            return
        elif self.builtin_exc_name == 'StopIteration' and (not self.exc_type):
            code.putln('%s = 1;' % Naming.error_without_exception_cname)
            code.putln('%s;' % code.error_goto(None))
            code.funcstate.error_without_exception = True
            return
        if self.exc_type:
            self.exc_type.generate_evaluation_code(code)
            type_code = self.exc_type.py_result()
            if self.exc_type.is_name:
                code.globalstate.use_entry_utility_code(self.exc_type.entry)
        else:
            type_code = '0'
        if self.exc_value:
            self.exc_value.generate_evaluation_code(code)
            value_code = self.exc_value.py_result()
        else:
            value_code = '0'
        if self.exc_tb:
            self.exc_tb.generate_evaluation_code(code)
            tb_code = self.exc_tb.py_result()
        else:
            tb_code = '0'
        if self.cause:
            self.cause.generate_evaluation_code(code)
            cause_code = self.cause.py_result()
        else:
            cause_code = '0'
        code.globalstate.use_utility_code(raise_utility_code)
        code.putln('__Pyx_Raise(%s, %s, %s, %s);' % (type_code, value_code, tb_code, cause_code))
        for obj in (self.exc_type, self.exc_value, self.exc_tb, self.cause):
            if obj:
                obj.generate_disposal_code(code)
                obj.free_temps(code)
        code.putln(code.error_goto(self.pos))

    def generate_function_definitions(self, env, code):
        if self.exc_type is not None:
            self.exc_type.generate_function_definitions(env, code)
        if self.exc_value is not None:
            self.exc_value.generate_function_definitions(env, code)
        if self.exc_tb is not None:
            self.exc_tb.generate_function_definitions(env, code)
        if self.cause is not None:
            self.cause.generate_function_definitions(env, code)

    def annotate(self, code):
        if self.exc_type:
            self.exc_type.annotate(code)
        if self.exc_value:
            self.exc_value.annotate(code)
        if self.exc_tb:
            self.exc_tb.annotate(code)
        if self.cause:
            self.cause.annotate(code)