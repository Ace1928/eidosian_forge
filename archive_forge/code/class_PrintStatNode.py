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
class PrintStatNode(StatNode):
    child_attrs = ['arg_tuple', 'stream']

    def analyse_expressions(self, env):
        if self.stream:
            stream = self.stream.analyse_expressions(env)
            self.stream = stream.coerce_to_pyobject(env)
        arg_tuple = self.arg_tuple.analyse_expressions(env)
        self.arg_tuple = arg_tuple.coerce_to_pyobject(env)
        env.use_utility_code(printing_utility_code)
        if len(self.arg_tuple.args) == 1 and self.append_newline:
            env.use_utility_code(printing_one_utility_code)
        return self
    nogil_check = Node.gil_error
    gil_message = 'Python print statement'

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        if self.stream:
            self.stream.generate_evaluation_code(code)
            stream_result = self.stream.py_result()
        else:
            stream_result = '0'
        if len(self.arg_tuple.args) == 1 and self.append_newline:
            arg = self.arg_tuple.args[0]
            arg.generate_evaluation_code(code)
            code.putln('if (__Pyx_PrintOne(%s, %s) < 0) %s' % (stream_result, arg.py_result(), code.error_goto(self.pos)))
            arg.generate_disposal_code(code)
            arg.free_temps(code)
        else:
            self.arg_tuple.generate_evaluation_code(code)
            code.putln('if (__Pyx_Print(%s, %s, %d) < 0) %s' % (stream_result, self.arg_tuple.py_result(), self.append_newline, code.error_goto(self.pos)))
            self.arg_tuple.generate_disposal_code(code)
            self.arg_tuple.free_temps(code)
        if self.stream:
            self.stream.generate_disposal_code(code)
            self.stream.free_temps(code)

    def generate_function_definitions(self, env, code):
        if self.stream:
            self.stream.generate_function_definitions(env, code)
        self.arg_tuple.generate_function_definitions(env, code)

    def annotate(self, code):
        if self.stream:
            self.stream.annotate(code)
        self.arg_tuple.annotate(code)