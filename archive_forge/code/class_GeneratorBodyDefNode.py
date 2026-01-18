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
class GeneratorBodyDefNode(DefNode):
    is_generator_body = True
    is_inlined = False
    is_async_gen_body = False
    inlined_comprehension_type = None

    def __init__(self, pos=None, name=None, body=None, is_async_gen_body=False):
        super(GeneratorBodyDefNode, self).__init__(pos=pos, body=body, name=name, is_async_gen_body=is_async_gen_body, doc=None, args=[], star_arg=None, starstar_arg=None)

    def declare_generator_body(self, env):
        prefix = env.next_id(env.scope_prefix)
        name = env.next_id('generator')
        cname = Naming.genbody_prefix + prefix + name
        entry = env.declare_var(None, py_object_type, self.pos, cname=cname, visibility='private')
        entry.func_cname = cname
        entry.qualified_name = EncodedString(self.name)
        entry.used = True
        self.entry = entry

    def analyse_declarations(self, env):
        self.analyse_argument_types(env)
        self.declare_generator_body(env)

    def generate_function_header(self, code, proto=False):
        header = 'static PyObject *%s(__pyx_CoroutineObject *%s, CYTHON_UNUSED PyThreadState *%s, PyObject *%s)' % (self.entry.func_cname, Naming.generator_cname, Naming.local_tstate_cname, Naming.sent_value_cname)
        if proto:
            code.putln('%s; /* proto */' % header)
        else:
            code.putln('%s /* generator body */\n{' % header)

    def generate_function_definitions(self, env, code):
        lenv = self.local_scope
        self.body.generate_function_definitions(lenv, code)
        code.enter_cfunc_scope(lenv)
        code.return_from_error_cleanup_label = code.new_label()
        code.mark_pos(self.pos)
        self.generate_cached_builtins_decls(lenv, code)
        code.putln('')
        self.generate_function_header(code)
        closure_init_code = code.insertion_point()
        code.putln('PyObject *%s = NULL;' % Naming.retval_cname)
        tempvardecl_code = code.insertion_point()
        code.put_declare_refcount_context()
        code.put_setup_refcount_context(self.entry.name or self.entry.qualified_name)
        profile = code.globalstate.directives['profile']
        linetrace = code.globalstate.directives['linetrace']
        if profile or linetrace:
            tempvardecl_code.put_trace_declarations()
            code.funcstate.can_trace = True
            code_object = self.code_object.calculate_result_code(code) if self.code_object else None
            code.put_trace_frame_init(code_object)
        code.funcstate.init_closure_temps(lenv.scope_class.type.scope)
        resume_code = code.insertion_point()
        first_run_label = code.new_label('first_run')
        code.use_label(first_run_label)
        code.put_label(first_run_label)
        code.putln('%s' % code.error_goto_if_null(Naming.sent_value_cname, self.pos))
        if self.is_inlined and self.inlined_comprehension_type is not None:
            target_type = self.inlined_comprehension_type
            if target_type is Builtin.list_type:
                comp_init = 'PyList_New(0)'
            elif target_type is Builtin.set_type:
                comp_init = 'PySet_New(NULL)'
            elif target_type is Builtin.dict_type:
                comp_init = 'PyDict_New()'
            else:
                raise InternalError('invalid type of inlined comprehension: %s' % target_type)
            code.putln('%s = %s; %s' % (Naming.retval_cname, comp_init, code.error_goto_if_null(Naming.retval_cname, self.pos)))
            code.put_gotref(Naming.retval_cname, py_object_type)
        self.generate_function_body(env, code)
        if lenv.scope_class.type.scope.var_entries:
            closure_init_code.putln('%s = %s;' % (lenv.scope_class.type.declaration_code(Naming.cur_scope_cname), lenv.scope_class.type.cast_code('%s->closure' % Naming.generator_cname)))
            code.putln('CYTHON_MAYBE_UNUSED_VAR(%s);' % Naming.cur_scope_cname)
        if profile or linetrace:
            code.funcstate.can_trace = False
        code.mark_pos(self.pos)
        code.putln('')
        code.putln('/* function exit code */')
        if not self.is_inlined and (not self.body.is_terminator):
            if self.is_async_gen_body:
                code.globalstate.use_utility_code(UtilityCode.load_cached('StopAsyncIteration', 'Coroutine.c'))
            code.putln('PyErr_SetNone(%s);' % ('__Pyx_PyExc_StopAsyncIteration' if self.is_async_gen_body else 'PyExc_StopIteration'))
        if code.label_used(code.error_label):
            if not self.body.is_terminator:
                code.put_goto(code.return_label)
            code.put_label(code.error_label)
            if self.is_inlined and self.inlined_comprehension_type is not None:
                code.put_xdecref_clear(Naming.retval_cname, py_object_type)
            if Future.generator_stop in env.global_scope().context.future_directives:
                code.globalstate.use_utility_code(UtilityCode.load_cached('pep479', 'Coroutine.c'))
                code.putln('__Pyx_Generator_Replace_StopIteration(%d);' % bool(self.is_async_gen_body))
            for cname, type in code.funcstate.all_managed_temps():
                code.put_xdecref(cname, type)
            code.put_add_traceback(self.entry.qualified_name)
        code.put_label(code.return_label)
        if self.is_inlined:
            code.put_xgiveref(Naming.retval_cname, py_object_type)
        else:
            code.put_xdecref_clear(Naming.retval_cname, py_object_type)
        code.putln('#if !CYTHON_USE_EXC_INFO_STACK')
        code.putln('__Pyx_Coroutine_ResetAndClearException(%s);' % Naming.generator_cname)
        code.putln('#endif')
        code.putln('%s->resume_label = -1;' % Naming.generator_cname)
        code.putln('__Pyx_Coroutine_clear((PyObject*)%s);' % Naming.generator_cname)
        if profile or linetrace:
            code.put_trace_return(Naming.retval_cname, nogil=not code.funcstate.gil_owned)
        code.put_finish_refcount_context()
        code.putln('return %s;' % Naming.retval_cname)
        code.putln('}')
        tempvardecl_code.put_temp_declarations(code.funcstate)
        if profile or linetrace:
            resume_code.put_trace_call(self.entry.qualified_name, self.pos, nogil=not code.funcstate.gil_owned)
        resume_code.putln('switch (%s->resume_label) {' % Naming.generator_cname)
        resume_code.putln('case 0: goto %s;' % first_run_label)
        for i, label in code.yield_labels:
            resume_code.putln('case %d: goto %s;' % (i, label))
        resume_code.putln('default: /* CPython raises the right error here */')
        if profile or linetrace:
            resume_code.put_trace_return('Py_None', nogil=not code.funcstate.gil_owned)
        resume_code.put_finish_refcount_context()
        resume_code.putln('return NULL;')
        resume_code.putln('}')
        code.exit_cfunc_scope()