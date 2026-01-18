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
class ExceptClauseNode(Node):
    child_attrs = ['pattern', 'target', 'body', 'exc_value']
    exc_value = None
    excinfo_target = None
    is_except_as = False

    def analyse_declarations(self, env):
        if self.target:
            self.target.analyse_target_declaration(env)
        self.body.analyse_declarations(env)

    def analyse_expressions(self, env):
        self.function_name = env.qualified_name
        if self.pattern:
            for i, pattern in enumerate(self.pattern):
                pattern = pattern.analyse_expressions(env)
                self.pattern[i] = pattern.coerce_to_pyobject(env)
        if self.target:
            from . import ExprNodes
            self.exc_value = ExprNodes.ExcValueNode(self.pos)
            self.target = self.target.analyse_target_expression(env, self.exc_value)
        self.body = self.body.analyse_expressions(env)
        return self

    def generate_handling_code(self, code, end_label):
        code.mark_pos(self.pos)
        if self.pattern:
            has_non_literals = not all((pattern.is_literal or (pattern.is_simple() and (not pattern.is_temp)) for pattern in self.pattern))
            if has_non_literals:
                exc_vars = [code.funcstate.allocate_temp(py_object_type, manage_ref=True) for _ in range(3)]
                code.globalstate.use_utility_code(UtilityCode.load_cached('PyErrFetchRestore', 'Exceptions.c'))
                code.putln('__Pyx_ErrFetch(&%s, &%s, &%s);' % tuple(exc_vars))
                exc_type = exc_vars[0]
            else:
                exc_vars = exc_type = None
            for pattern in self.pattern:
                pattern.generate_evaluation_code(code)
            patterns = [pattern.py_result() for pattern in self.pattern]
            exc_tests = []
            if exc_type:
                code.globalstate.use_utility_code(UtilityCode.load_cached('FastTypeChecks', 'ModuleSetupCode.c'))
                if len(patterns) == 2:
                    exc_tests.append('__Pyx_PyErr_GivenExceptionMatches2(%s, %s, %s)' % (exc_type, patterns[0], patterns[1]))
                else:
                    exc_tests.extend(('__Pyx_PyErr_GivenExceptionMatches(%s, %s)' % (exc_type, pattern) for pattern in patterns))
            elif len(patterns) == 2:
                code.globalstate.use_utility_code(UtilityCode.load_cached('FastTypeChecks', 'ModuleSetupCode.c'))
                exc_tests.append('__Pyx_PyErr_ExceptionMatches2(%s, %s)' % (patterns[0], patterns[1]))
            else:
                code.globalstate.use_utility_code(UtilityCode.load_cached('PyErrExceptionMatches', 'Exceptions.c'))
                exc_tests.extend(('__Pyx_PyErr_ExceptionMatches(%s)' % pattern for pattern in patterns))
            match_flag = code.funcstate.allocate_temp(PyrexTypes.c_int_type, manage_ref=False)
            code.putln('%s = %s;' % (match_flag, ' || '.join(exc_tests)))
            for pattern in self.pattern:
                pattern.generate_disposal_code(code)
                pattern.free_temps(code)
            if exc_vars:
                code.putln('__Pyx_ErrRestore(%s, %s, %s);' % tuple(exc_vars))
                code.putln(' '.join(['%s = 0;' % var for var in exc_vars]))
                for temp in exc_vars:
                    code.funcstate.release_temp(temp)
            code.putln('if (%s) {' % match_flag)
            code.funcstate.release_temp(match_flag)
        else:
            code.putln('/*except:*/ {')
        if not getattr(self.body, 'stats', True) and self.excinfo_target is None and (self.target is None):
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyErrFetchRestore', 'Exceptions.c'))
            code.putln('__Pyx_ErrRestore(0,0,0);')
            code.put_goto(end_label)
            code.putln('}')
            return
        exc_vars = [code.funcstate.allocate_temp(py_object_type, manage_ref=True) for _ in range(3)]
        code.put_add_traceback(self.function_name)
        code.globalstate.use_utility_code(get_exception_utility_code)
        exc_args = '&%s, &%s, &%s' % tuple(exc_vars)
        code.putln('if (__Pyx_GetException(%s) < 0) %s' % (exc_args, code.error_goto(self.pos)))
        for var in exc_vars:
            code.put_xgotref(var, py_object_type)
        if self.target:
            self.exc_value.set_var(exc_vars[1])
            self.exc_value.generate_evaluation_code(code)
            self.target.generate_assignment_code(self.exc_value, code)
        if self.excinfo_target is not None:
            for tempvar, node in zip(exc_vars, self.excinfo_target.args):
                node.set_var(tempvar)
        old_loop_labels = code.new_loop_labels('except_')
        old_exc_vars = code.funcstate.exc_vars
        code.funcstate.exc_vars = exc_vars
        self.body.generate_execution_code(code)
        code.funcstate.exc_vars = old_exc_vars
        if not self.body.is_terminator:
            for var in exc_vars:
                code.put_xdecref_clear(var, py_object_type)
            code.put_goto(end_label)
        for _ in code.label_interceptor(code.get_loop_labels(), old_loop_labels):
            for i, var in enumerate(exc_vars):
                (code.put_decref_clear if i < 2 else code.put_xdecref_clear)(var, py_object_type)
        code.set_loop_labels(old_loop_labels)
        for temp in exc_vars:
            code.funcstate.release_temp(temp)
        code.putln('}')

    def generate_function_definitions(self, env, code):
        if self.target is not None:
            self.target.generate_function_definitions(env, code)
        self.body.generate_function_definitions(env, code)

    def annotate(self, code):
        if self.pattern:
            for pattern in self.pattern:
                pattern.annotate(code)
        if self.target:
            self.target.annotate(code)
        self.body.annotate(code)