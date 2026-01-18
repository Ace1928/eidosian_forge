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
class FuncDefNode(StatNode, BlockNode):
    py_func = None
    needs_closure = False
    needs_outer_scope = False
    pymethdef_required = False
    is_generator = False
    is_generator_expression = False
    is_coroutine = False
    is_asyncgen = False
    is_generator_body = False
    is_async_def = False
    modifiers = []
    has_fused_arguments = False
    star_arg = None
    starstar_arg = None
    is_cyfunction = False
    code_object = None
    return_type_annotation = None
    outer_attrs = None

    def analyse_default_values(self, env):
        default_seen = 0
        for arg in self.args:
            if arg.default:
                default_seen = 1
                if arg.is_generic:
                    arg.default = arg.default.analyse_types(env)
                    arg.default = arg.default.coerce_to(arg.type, env)
                elif arg.is_special_method_optional:
                    if not arg.default.is_none:
                        error(arg.pos, 'This argument cannot have a non-None default value')
                        arg.default = None
                else:
                    error(arg.pos, 'This argument cannot have a default value')
                    arg.default = None
            elif arg.kw_only:
                default_seen = 1
            elif default_seen:
                error(arg.pos, 'Non-default argument following default argument')

    def analyse_annotations(self, env):
        for arg in self.args:
            if arg.annotation:
                arg.annotation = arg.annotation.analyse_types(env)
        if self.return_type_annotation:
            self.return_type_annotation = self.return_type_annotation.analyse_types(env)

    def align_argument_type(self, env, arg):
        directive_locals = self.directive_locals
        orig_type = arg.type
        if arg.name in directive_locals:
            type_node = directive_locals[arg.name]
            other_type = type_node.analyse_as_type(env)
        elif isinstance(arg, CArgDeclNode) and arg.annotation and env.directives['annotation_typing']:
            type_node = arg.annotation
            other_type = arg.inject_type_from_annotations(env)
            if other_type is None:
                return arg
        else:
            return arg
        if other_type is None:
            error(type_node.pos, 'Not a type')
        elif orig_type is not py_object_type and (not orig_type.same_as(other_type)):
            error(arg.base_type.pos, 'Signature does not agree with previous declaration')
            error(type_node.pos, 'Previous declaration here')
        else:
            arg.type = other_type
            if arg.type.is_complex:
                arg.type.create_declaration_utility_code(env)
        return arg

    def need_gil_acquisition(self, lenv):
        return 0

    def create_local_scope(self, env):
        genv = env
        while genv.is_py_class_scope or genv.is_c_class_scope:
            genv = genv.outer_scope
        if self.needs_closure:
            cls = GeneratorExpressionScope if self.is_generator_expression else ClosureScope
            lenv = cls(name=self.entry.name, outer_scope=genv, parent_scope=env, scope_name=self.entry.cname)
        else:
            lenv = LocalScope(name=self.entry.name, outer_scope=genv, parent_scope=env)
        lenv.return_type = self.return_type
        type = self.entry.type
        if type.is_cfunction:
            lenv.nogil = type.nogil and (not type.with_gil)
        self.local_scope = lenv
        lenv.directives = env.directives
        return lenv

    def generate_function_body(self, env, code):
        self.body.generate_execution_code(code)

    def generate_function_definitions(self, env, code):
        from . import Buffer
        lenv = self.local_scope
        if lenv.is_closure_scope and (not lenv.is_passthrough):
            outer_scope_cname = '%s->%s' % (Naming.cur_scope_cname, Naming.outer_scope_cname)
        else:
            outer_scope_cname = Naming.outer_scope_cname
        lenv.mangle_closure_cnames(outer_scope_cname)
        self.body.generate_function_definitions(lenv, code)
        self.generate_lambda_definitions(lenv, code)
        is_getbuffer_slot = self.entry.name == '__getbuffer__' and self.entry.scope.is_c_class_scope
        is_releasebuffer_slot = self.entry.name == '__releasebuffer__' and self.entry.scope.is_c_class_scope
        is_buffer_slot = is_getbuffer_slot or is_releasebuffer_slot
        if is_buffer_slot:
            if 'cython_unused' not in self.modifiers:
                self.modifiers = self.modifiers + ['cython_unused']
        preprocessor_guard = self.get_preprocessor_guard()
        profile = code.globalstate.directives['profile']
        linetrace = code.globalstate.directives['linetrace']
        if profile or linetrace:
            if linetrace:
                code.use_fast_gil_utility_code()
            code.globalstate.use_utility_code(UtilityCode.load_cached('Profile', 'Profile.c'))
        code.enter_cfunc_scope(lenv)
        code.return_from_error_cleanup_label = code.new_label()
        code.funcstate.gil_owned = not lenv.nogil
        code.mark_pos(self.pos)
        self.generate_cached_builtins_decls(lenv, code)
        code.putln('')
        if preprocessor_guard:
            code.putln(preprocessor_guard)
        with_pymethdef = self.needs_assignment_synthesis(env, code) or self.pymethdef_required
        if self.py_func:
            self.py_func.generate_function_header(code, with_pymethdef=with_pymethdef, proto_only=True)
        self.generate_function_header(code, with_pymethdef=with_pymethdef)
        cenv = env
        while cenv.is_py_class_scope or cenv.is_c_class_scope:
            cenv = cenv.outer_scope
        if self.needs_closure:
            code.put(lenv.scope_class.type.declaration_code(Naming.cur_scope_cname))
            code.putln(';')
        elif self.needs_outer_scope:
            if lenv.is_passthrough:
                code.put(lenv.scope_class.type.declaration_code(Naming.cur_scope_cname))
                code.putln(';')
            code.put(cenv.scope_class.type.declaration_code(Naming.outer_scope_cname))
            code.putln(';')
        self.generate_argument_declarations(lenv, code)
        for entry in lenv.var_entries:
            if not (entry.in_closure or entry.is_arg):
                code.put_var_declaration(entry)
        init = ''
        return_type = self.return_type
        if return_type.is_cv_qualified and return_type.is_const:
            return_type = return_type.cv_base_type
        if not return_type.is_void:
            if return_type.is_pyobject:
                init = ' = NULL'
            elif return_type.is_memoryviewslice:
                init = ' = ' + return_type.literal_code(return_type.default_value)
            code.putln('%s%s;' % (return_type.declaration_code(Naming.retval_cname), init))
        tempvardecl_code = code.insertion_point()
        self.generate_keyword_list(code)
        acquire_gil = self.acquire_gil
        used_buffer_entries = [entry for entry in lenv.buffer_entries if entry.used]
        var_decls_definitely_need_gil = lenv.nogil and (self.needs_closure or self.needs_outer_scope)
        gilstate_decl = None
        var_decls_need_gil = False
        if acquire_gil or var_decls_definitely_need_gil:
            code.put_ensure_gil()
            code.funcstate.gil_owned = True
            var_decls_need_gil = True
        else:
            gilstate_decl = code.insertion_point()
        if profile or linetrace:
            if not self.is_generator:
                tempvardecl_code.put_trace_declarations()
                code_object = self.code_object.calculate_result_code(code) if self.code_object else None
                code.put_trace_frame_init(code_object)
        if is_getbuffer_slot:
            self.getbuffer_check(code)
        refnanny_decl_code = tempvardecl_code.insertion_point()
        refnanny_setup_code = code.insertion_point()
        if is_getbuffer_slot:
            self.getbuffer_init(code)
        if self.needs_closure:
            tp_slot = TypeSlots.ConstructorSlot('tp_new', '__new__')
            slot_func_cname = TypeSlots.get_slot_function(lenv.scope_class.type.scope, tp_slot)
            if not slot_func_cname:
                slot_func_cname = '%s->tp_new' % lenv.scope_class.type.typeptr_cname
            code.putln('%s = (%s)%s(%s, %s, NULL);' % (Naming.cur_scope_cname, lenv.scope_class.type.empty_declaration_code(), slot_func_cname, lenv.scope_class.type.typeptr_cname, Naming.empty_tuple))
            code.putln('if (unlikely(!%s)) {' % Naming.cur_scope_cname)
            code.putln('%s = %s;' % (Naming.cur_scope_cname, lenv.scope_class.type.cast_code('Py_None')))
            code.put_incref('Py_None', py_object_type)
            code.putln(code.error_goto(self.pos))
            code.putln('} else {')
            code.put_gotref(Naming.cur_scope_cname, lenv.scope_class.type)
            code.putln('}')
        if self.needs_outer_scope:
            if self.is_cyfunction:
                code.putln('%s = (%s) __Pyx_CyFunction_GetClosure(%s);' % (outer_scope_cname, cenv.scope_class.type.empty_declaration_code(), Naming.self_cname))
            else:
                code.putln('%s = (%s) %s;' % (outer_scope_cname, cenv.scope_class.type.empty_declaration_code(), Naming.self_cname))
            if lenv.is_passthrough:
                code.putln('%s = %s;' % (Naming.cur_scope_cname, outer_scope_cname))
            elif self.needs_closure:
                code.put_incref(outer_scope_cname, cenv.scope_class.type)
                code.put_giveref(outer_scope_cname, cenv.scope_class.type)
        if profile or linetrace:
            if not self.is_generator:
                if self.is_wrapper:
                    trace_name = self.entry.name + ' (wrapper)'
                else:
                    trace_name = self.entry.name
                code.put_trace_call(trace_name, self.pos, nogil=not code.funcstate.gil_owned)
            code.funcstate.can_trace = True
        self.generate_argument_parsing_code(env, code)
        for entry in lenv.arg_entries:
            if not entry.type.is_memoryviewslice:
                if (acquire_gil or entry.cf_is_reassigned) and (not entry.in_closure):
                    code.put_var_incref(entry)
            elif entry.cf_is_reassigned and (not entry.in_closure):
                code.put_var_incref_memoryviewslice(entry, have_gil=code.funcstate.gil_owned)
        for entry in lenv.var_entries:
            if entry.is_arg and entry.cf_is_reassigned and (not entry.in_closure):
                if entry.type.is_memoryviewslice:
                    code.put_var_incref_memoryviewslice(entry, have_gil=code.funcstate.gil_owned)
                if entry.xdecref_cleanup:
                    code.put_var_xincref(entry)
                else:
                    code.put_var_incref(entry)
        for entry in lenv.var_entries + lenv.arg_entries:
            if entry.type.is_buffer and entry.buffer_aux.buflocal_nd_var.used:
                Buffer.put_init_vars(entry, code)
        self.generate_argument_type_tests(code)
        for entry in lenv.arg_entries:
            if entry.type.is_buffer:
                Buffer.put_acquire_arg_buffer(entry, code, self.pos)
        if code.funcstate.needs_refnanny:
            var_decls_need_gil = True
        if var_decls_need_gil and lenv.nogil:
            if gilstate_decl is not None:
                gilstate_decl.put_ensure_gil()
                gilstate_decl = None
                code.funcstate.gil_owned = True
            code.put_release_ensured_gil()
            code.funcstate.gil_owned = False
        self.generate_function_body(env, code)
        code.mark_pos(self.pos, trace=False)
        code.putln('')
        code.putln('/* function exit code */')
        gil_owned = {'success': code.funcstate.gil_owned, 'error': code.funcstate.gil_owned, 'gil_state_declared': gilstate_decl is None}

        def assure_gil(code_path, code=code):
            if not gil_owned[code_path]:
                if not gil_owned['gil_state_declared']:
                    gilstate_decl.declare_gilstate()
                    gil_owned['gil_state_declared'] = True
                code.put_ensure_gil(declare_gilstate=False)
                gil_owned[code_path] = True
        return_type = self.return_type
        if not self.body.is_terminator:
            if return_type.is_pyobject:
                lhs = Naming.retval_cname
                assure_gil('success')
                code.put_init_to_py_none(lhs, return_type)
            elif not return_type.is_memoryviewslice:
                val = return_type.default_value
                if val:
                    code.putln('%s = %s;' % (Naming.retval_cname, val))
                elif not return_type.is_void:
                    code.putln('__Pyx_pretend_to_initialize(&%s);' % Naming.retval_cname)
        if code.label_used(code.error_label):
            if not self.body.is_terminator:
                code.put_goto(code.return_label)
            code.put_label(code.error_label)
            for cname, type in code.funcstate.all_managed_temps():
                assure_gil('error')
                code.put_xdecref(cname, type, have_gil=gil_owned['error'])
            buffers_present = len(used_buffer_entries) > 0
            if buffers_present:
                code.globalstate.use_utility_code(restore_exception_utility_code)
                code.putln('{ PyObject *__pyx_type, *__pyx_value, *__pyx_tb;')
                code.putln('__Pyx_PyThreadState_declare')
                assure_gil('error')
                code.putln('__Pyx_PyThreadState_assign')
                code.putln('__Pyx_ErrFetch(&__pyx_type, &__pyx_value, &__pyx_tb);')
                for entry in used_buffer_entries:
                    Buffer.put_release_buffer_code(code, entry)
                code.putln('__Pyx_ErrRestore(__pyx_type, __pyx_value, __pyx_tb);}')
            if return_type.is_memoryviewslice:
                from . import MemoryView
                MemoryView.put_init_entry(Naming.retval_cname, code)
                err_val = Naming.retval_cname
            else:
                err_val = self.error_value()
            exc_check = self.caller_will_check_exceptions()
            if err_val is not None or exc_check:
                assure_gil('error')
                if code.funcstate.error_without_exception:
                    tempvardecl_code.putln('int %s = 0; /* StopIteration */' % Naming.error_without_exception_cname)
                    code.putln('if (!%s) {' % Naming.error_without_exception_cname)
                code.put_add_traceback(self.entry.qualified_name)
                if code.funcstate.error_without_exception:
                    code.putln('}')
            else:
                warning(self.entry.pos, "Unraisable exception in function '%s'." % self.entry.qualified_name, 0)
                assure_gil('error')
                code.put_unraisable(self.entry.qualified_name)
            default_retval = return_type.default_value
            if err_val is None and default_retval:
                err_val = default_retval
            if err_val is not None:
                if err_val != Naming.retval_cname:
                    code.putln('%s = %s;' % (Naming.retval_cname, err_val))
            elif not return_type.is_void:
                code.putln('__Pyx_pretend_to_initialize(&%s);' % Naming.retval_cname)
            if is_getbuffer_slot:
                assure_gil('error')
                self.getbuffer_error_cleanup(code)

            def align_error_path_gil_to_success_path(code=code.insertion_point()):
                if gil_owned['success']:
                    assure_gil('error', code=code)
                elif gil_owned['error']:
                    code.put_release_ensured_gil()
                    gil_owned['error'] = False
                assert gil_owned['error'] == gil_owned['success'], '%s: error path %s != success path %s' % (self.pos, gil_owned['error'], gil_owned['success'])
            if buffers_present or is_getbuffer_slot or return_type.is_memoryviewslice:
                assert gil_owned['error'] or return_type.is_memoryviewslice
                code.put_goto(code.return_from_error_cleanup_label)
            else:
                align_error_path_gil_to_success_path()
        else:

            def align_error_path_gil_to_success_path():
                pass
        if code.label_used(code.return_label) or not code.label_used(code.error_label):
            code.put_label(code.return_label)
            for entry in used_buffer_entries:
                assure_gil('success')
                Buffer.put_release_buffer_code(code, entry)
            if is_getbuffer_slot:
                assure_gil('success')
                self.getbuffer_normal_cleanup(code)
            if return_type.is_memoryviewslice:
                cond = code.unlikely(return_type.error_condition(Naming.retval_cname))
                code.putln('if (%s) {' % cond)
                if not gil_owned['success']:
                    code.put_ensure_gil()
                code.putln('PyErr_SetString(PyExc_TypeError, "Memoryview return value is not initialized");')
                if not gil_owned['success']:
                    code.put_release_ensured_gil()
                code.putln('}')
        if code.label_used(code.return_from_error_cleanup_label):
            align_error_path_gil_to_success_path()
            code.put_label(code.return_from_error_cleanup_label)
        for entry in lenv.var_entries:
            if not entry.used or entry.in_closure:
                continue
            if entry.type.needs_refcounting:
                if entry.is_arg and (not entry.cf_is_reassigned):
                    continue
                if entry.type.refcounting_needs_gil:
                    assure_gil('success')
            code.put_var_xdecref(entry, have_gil=gil_owned['success'])
        for entry in lenv.arg_entries:
            if entry.in_closure:
                continue
            if entry.type.is_memoryviewslice:
                if not entry.cf_is_reassigned:
                    continue
            else:
                if not acquire_gil and (not entry.cf_is_reassigned):
                    continue
                if entry.type.needs_refcounting:
                    assure_gil('success')
            code.put_var_xdecref(entry, have_gil=gil_owned['success'])
        if self.needs_closure:
            assure_gil('success')
            code.put_decref(Naming.cur_scope_cname, lenv.scope_class.type)
        if not lenv.nogil:
            default_retval = return_type.default_value
            err_val = self.error_value()
            if err_val is None and default_retval:
                err_val = default_retval
            code.put_xgiveref(Naming.retval_cname, return_type)
        if self.entry.is_special and self.entry.name == '__hash__':
            assure_gil('success')
            code.putln('if (unlikely(%s == -1) && !PyErr_Occurred()) %s = -2;' % (Naming.retval_cname, Naming.retval_cname))
        if profile or linetrace:
            code.funcstate.can_trace = False
            if not self.is_generator:
                if return_type.is_pyobject:
                    code.put_trace_return(Naming.retval_cname, nogil=not gil_owned['success'])
                else:
                    code.put_trace_return('Py_None', nogil=not gil_owned['success'])
        if code.funcstate.needs_refnanny:
            refnanny_decl_code.put_declare_refcount_context()
            refnanny_setup_code.put_setup_refcount_context(self.entry.name, acquire_gil=not var_decls_need_gil)
            code.put_finish_refcount_context(nogil=not gil_owned['success'])
        if acquire_gil or (lenv.nogil and gil_owned['success']):
            code.put_release_ensured_gil()
            code.funcstate.gil_owned = False
        if not return_type.is_void:
            code.putln('return %s;' % Naming.retval_cname)
        code.putln('}')
        if preprocessor_guard:
            code.putln('#endif /*!(%s)*/' % preprocessor_guard)
        tempvardecl_code.put_temp_declarations(code.funcstate)
        code.exit_cfunc_scope()
        if self.py_func:
            self.py_func.generate_function_definitions(env, code)
        self.generate_wrapper_functions(code)

    def declare_argument(self, env, arg):
        if arg.type.is_void:
            error(arg.pos, "Invalid use of 'void'")
        elif not arg.type.is_complete() and (not (arg.type.is_array or arg.type.is_memoryviewslice)):
            error(arg.pos, "Argument type '%s' is incomplete" % arg.type)
        entry = env.declare_arg(arg.name, arg.type, arg.pos)
        if arg.annotation:
            entry.annotation = arg.annotation
        return entry

    def generate_arg_type_test(self, arg, code):
        if arg.type.typeobj_is_available():
            code.globalstate.use_utility_code(UtilityCode.load_cached('ArgTypeTest', 'FunctionArguments.c'))
            typeptr_cname = arg.type.typeptr_cname
            arg_code = '((PyObject *)%s)' % arg.entry.cname
            code.putln('if (unlikely(!__Pyx_ArgTypeTest(%s, %s, %d, %s, %s))) %s' % (arg_code, typeptr_cname, arg.accept_none, arg.name_cstring, arg.type.is_builtin_type and arg.type.require_exact, code.error_goto(arg.pos)))
        else:
            error(arg.pos, 'Cannot test type of extern C class without type object name specification')

    def generate_arg_none_check(self, arg, code):
        if arg.type.is_memoryviewslice:
            cname = '%s.memview' % arg.entry.cname
        else:
            cname = arg.entry.cname
        code.putln('if (unlikely(((PyObject *)%s) == Py_None)) {' % cname)
        code.putln('PyErr_Format(PyExc_TypeError, "Argument \'%%.%ds\' must not be None", %s); %s' % (max(200, len(arg.name_cstring)), arg.name_cstring, code.error_goto(arg.pos)))
        code.putln('}')

    def generate_wrapper_functions(self, code):
        pass

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        if not self.is_wrapper:
            for arg in self.args:
                if not arg.is_dynamic:
                    arg.generate_assignment_code(code)

    def _get_py_buffer_info(self):
        py_buffer = self.local_scope.arg_entries[1]
        try:
            obj_type = py_buffer.type.base_type.scope.entries['obj'].type
        except (AttributeError, KeyError):
            obj_type = None
        return (py_buffer, obj_type)

    def getbuffer_check(self, code):
        py_buffer, _ = self._get_py_buffer_info()
        view = py_buffer.cname
        code.putln('if (unlikely(%s == NULL)) {' % view)
        code.putln('PyErr_SetString(PyExc_BufferError, "PyObject_GetBuffer: view==NULL argument is obsolete");')
        code.putln('return -1;')
        code.putln('}')

    def getbuffer_init(self, code):
        py_buffer, obj_type = self._get_py_buffer_info()
        view = py_buffer.cname
        if obj_type and obj_type.is_pyobject:
            code.put_init_to_py_none('%s->obj' % view, obj_type)
            code.put_giveref('%s->obj' % view, obj_type)
        else:
            code.putln('%s->obj = NULL;' % view)

    def getbuffer_error_cleanup(self, code):
        py_buffer, obj_type = self._get_py_buffer_info()
        view = py_buffer.cname
        if obj_type and obj_type.is_pyobject:
            code.putln('if (%s->obj != NULL) {' % view)
            code.put_gotref('%s->obj' % view, obj_type)
            code.put_decref_clear('%s->obj' % view, obj_type)
            code.putln('}')
        else:
            code.putln('Py_CLEAR(%s->obj);' % view)

    def getbuffer_normal_cleanup(self, code):
        py_buffer, obj_type = self._get_py_buffer_info()
        view = py_buffer.cname
        if obj_type and obj_type.is_pyobject:
            code.putln('if (%s->obj == Py_None) {' % view)
            code.put_gotref('%s->obj' % view, obj_type)
            code.put_decref_clear('%s->obj' % view, obj_type)
            code.putln('}')

    def get_preprocessor_guard(self):
        if not self.entry.is_special:
            return None
        name = self.entry.name
        slot = TypeSlots.get_slot_table(self.local_scope.directives).get_slot_by_method_name(name)
        if not slot:
            return None
        if name == '__long__' and (not self.entry.scope.lookup_here('__int__')):
            return None
        if name in ('__getbuffer__', '__releasebuffer__') and self.entry.scope.is_c_class_scope:
            return None
        return slot.preprocessor_guard_code()