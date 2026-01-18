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
class DefNodeWrapper(FuncDefNode):
    defnode = None
    target = None
    needs_values_cleanup = False

    def __init__(self, *args, **kwargs):
        FuncDefNode.__init__(self, *args, **kwargs)
        self.num_posonly_args = self.target.num_posonly_args
        self.num_kwonly_args = self.target.num_kwonly_args
        self.num_required_kw_args = self.target.num_required_kw_args
        self.num_required_args = self.target.num_required_args
        self.self_in_stararg = self.target.self_in_stararg
        self.signature = None

    def analyse_declarations(self, env):
        target_entry = self.target.entry
        name = self.name
        prefix = env.next_id(env.scope_prefix)
        target_entry.func_cname = punycodify_name(Naming.pywrap_prefix + prefix + name)
        target_entry.pymethdef_cname = punycodify_name(Naming.pymethdef_prefix + prefix + name)
        self.signature = target_entry.signature
        self.np_args_idx = self.target.np_args_idx

    def prepare_argument_coercion(self, env):
        for arg in self.args:
            if not arg.type.is_pyobject:
                if not arg.type.create_from_py_utility_code(env):
                    pass
            elif arg.hdr_type and (not arg.hdr_type.is_pyobject):
                if not arg.hdr_type.create_to_py_utility_code(env):
                    pass
        if self.starstar_arg and (not self.starstar_arg.entry.cf_used):
            entry = self.starstar_arg.entry
            entry.xdecref_cleanup = 1
            for ass in entry.cf_assignments:
                if not ass.is_arg and ass.lhs.is_name:
                    ass.lhs.cf_maybe_null = True

    def signature_has_nongeneric_args(self):
        argcount = len(self.args)
        if argcount == 0 or (argcount == 1 and (self.args[0].is_self_arg or self.args[0].is_type_arg)):
            return 0
        return 1

    def signature_has_generic_args(self):
        return self.signature.has_generic_args

    def generate_function_body(self, code):
        args = []
        if self.signature.has_dummy_arg:
            args.append(Naming.self_cname)
        for arg in self.args:
            if arg.type.is_cpp_class:
                code.globalstate.use_utility_code(UtilityCode.load_cached('MoveIfSupported', 'CppSupport.cpp'))
                args.append('__PYX_STD_MOVE_IF_SUPPORTED(%s)' % arg.entry.cname)
            elif arg.hdr_type and (not (arg.type.is_memoryviewslice or arg.type.is_struct or arg.type.is_complex)):
                args.append(arg.type.cast_code(arg.entry.cname))
            else:
                args.append(arg.entry.cname)
        if self.star_arg:
            args.append(self.star_arg.entry.cname)
        if self.starstar_arg:
            args.append(self.starstar_arg.entry.cname)
        args = ', '.join(args)
        if not self.return_type.is_void:
            code.put('%s = ' % Naming.retval_cname)
        code.putln('%s(%s);' % (self.target.entry.pyfunc_cname, args))

    def generate_function_definitions(self, env, code):
        lenv = self.target.local_scope
        code.mark_pos(self.pos)
        code.putln('')
        code.putln('/* Python wrapper */')
        preprocessor_guard = self.target.get_preprocessor_guard()
        if preprocessor_guard:
            code.putln(preprocessor_guard)
        code.enter_cfunc_scope(lenv)
        code.return_from_error_cleanup_label = code.new_label()
        with_pymethdef = self.target.needs_assignment_synthesis(env, code) or self.target.pymethdef_required
        self.generate_function_header(code, with_pymethdef)
        self.generate_argument_declarations(lenv, code)
        tempvardecl_code = code.insertion_point()
        if self.return_type.is_pyobject:
            retval_init = ' = 0'
        else:
            retval_init = ''
        if not self.return_type.is_void:
            code.putln('%s%s;' % (self.return_type.declaration_code(Naming.retval_cname), retval_init))
        code.put_declare_refcount_context()
        code.put_setup_refcount_context(EncodedString('%s (wrapper)' % self.name))
        self.generate_argument_parsing_code(lenv, code, tempvardecl_code)
        self.generate_argument_type_tests(code)
        self.generate_function_body(code)
        tempvardecl_code.put_temp_declarations(code.funcstate)
        code.mark_pos(self.pos)
        code.putln('')
        code.putln('/* function exit code */')
        if code.error_label in code.labels_used:
            code.put_goto(code.return_label)
            code.put_label(code.error_label)
            for cname, type in code.funcstate.all_managed_temps():
                code.put_xdecref(cname, type)
            err_val = self.error_value()
            if err_val is not None:
                code.putln('%s = %s;' % (Naming.retval_cname, err_val))
        code.put_label(code.return_label)
        for entry in lenv.var_entries:
            if entry.is_arg:
                if entry.xdecref_cleanup:
                    code.put_var_xdecref(entry)
                else:
                    code.put_var_decref(entry)
        var_entries_set = set(lenv.var_entries)
        for arg in self.args:
            if not arg.type.is_pyobject and arg.entry not in var_entries_set:
                if arg.entry.xdecref_cleanup:
                    code.put_var_xdecref(arg.entry)
                else:
                    code.put_var_decref(arg.entry)
        self.generate_argument_values_cleanup_code(code)
        code.put_finish_refcount_context()
        if not self.return_type.is_void:
            code.putln('return %s;' % Naming.retval_cname)
        code.putln('}')
        code.exit_cfunc_scope()
        if preprocessor_guard:
            code.putln('#endif /*!(%s)*/' % preprocessor_guard)

    def generate_function_header(self, code, with_pymethdef, proto_only=0):
        arg_code_list = []
        sig = self.signature
        if sig.has_dummy_arg or self.self_in_stararg:
            arg_code = 'PyObject *%s' % Naming.self_cname
            if not sig.has_dummy_arg:
                arg_code = 'CYTHON_UNUSED ' + arg_code
            arg_code_list.append(arg_code)
        for arg in self.args:
            if not arg.is_generic:
                if arg.is_self_arg or arg.is_type_arg:
                    arg_code_list.append('PyObject *%s' % arg.hdr_cname)
                else:
                    arg_code_list.append(arg.hdr_type.declaration_code(arg.hdr_cname))
        entry = self.target.entry
        if not entry.is_special and sig.method_flags() == [TypeSlots.method_noargs]:
            arg_code_list.append('CYTHON_UNUSED PyObject *unused')
        if sig.has_generic_args:
            varargs_args = 'PyObject *%s, PyObject *%s' % (Naming.args_cname, Naming.kwds_cname)
            if sig.use_fastcall:
                fastcall_args = 'PyObject *const *%s, Py_ssize_t %s, PyObject *%s' % (Naming.args_cname, Naming.nargs_cname, Naming.kwds_cname)
                arg_code_list.append('\n#if CYTHON_METH_FASTCALL\n%s\n#else\n%s\n#endif\n' % (fastcall_args, varargs_args))
            else:
                arg_code_list.append(varargs_args)
        if entry.is_special:
            for n in range(len(self.args), sig.max_num_fixed_args()):
                arg_code_list.append('CYTHON_UNUSED PyObject *unused_arg_%s' % n)
        arg_code = ', '.join(arg_code_list)
        mf = ''
        if entry.name in ('__getbuffer__', '__releasebuffer__') and entry.scope.is_c_class_scope:
            mf = 'CYTHON_UNUSED '
            with_pymethdef = False
        dc = self.return_type.declaration_code(entry.func_cname)
        header = '%sstatic %s(%s)' % (mf, dc, arg_code)
        code.putln('%s; /*proto*/' % header)
        if proto_only:
            if self.target.fused_py_func:
                self.target.fused_py_func.generate_function_header(code, with_pymethdef, proto_only=True)
            return
        if Options.docstrings and entry.doc and (not self.target.fused_py_func) and (not entry.scope.is_property_scope) and (not entry.is_special or entry.wrapperbase_cname):
            docstr = entry.doc
            if docstr.is_unicode:
                docstr = docstr.as_utf8_string()
            if not (entry.is_special and entry.name in ('__getbuffer__', '__releasebuffer__')):
                code.putln('PyDoc_STRVAR(%s, %s);' % (entry.doc_cname, docstr.as_c_string_literal()))
            if entry.is_special:
                code.putln('#if CYTHON_UPDATE_DESCRIPTOR_DOC')
                code.putln('struct wrapperbase %s;' % entry.wrapperbase_cname)
                code.putln('#endif')
        if with_pymethdef or self.target.fused_py_func:
            code.put('static PyMethodDef %s = ' % entry.pymethdef_cname)
            code.put_pymethoddef(self.target.entry, ';', allow_skip=False)
        code.putln('%s {' % header)

    def generate_argument_declarations(self, env, code):
        for arg in self.args:
            if arg.is_generic:
                if arg.needs_conversion:
                    code.putln('PyObject *%s = 0;' % arg.hdr_cname)
                else:
                    code.put_var_declaration(arg.entry)
        for entry in env.var_entries:
            if entry.is_arg:
                code.put_var_declaration(entry)
        if self.signature_has_generic_args():
            nargs_code = 'CYTHON_UNUSED Py_ssize_t %s;' % Naming.nargs_cname
            if self.signature.use_fastcall:
                code.putln('#if !CYTHON_METH_FASTCALL')
                code.putln(nargs_code)
                code.putln('#endif')
            else:
                code.putln(nargs_code)
        code.putln('CYTHON_UNUSED PyObject *const *%s;' % Naming.kwvalues_cname)

    def generate_argument_parsing_code(self, env, code, decl_code):
        old_error_label = code.new_error_label()
        our_error_label = code.error_label
        end_label = code.new_label('argument_unpacking_done')
        has_kwonly_args = self.num_kwonly_args > 0
        has_star_or_kw_args = self.star_arg is not None or self.starstar_arg is not None or has_kwonly_args
        for arg in self.args:
            if not arg.type.is_pyobject:
                if not arg.type.create_from_py_utility_code(env):
                    pass
        if self.signature_has_generic_args():
            if self.signature.use_fastcall:
                code.putln('#if !CYTHON_METH_FASTCALL')
            code.putln('#if CYTHON_ASSUME_SAFE_MACROS')
            code.putln('%s = PyTuple_GET_SIZE(%s);' % (Naming.nargs_cname, Naming.args_cname))
            code.putln('#else')
            code.putln('%s = PyTuple_Size(%s); if (%s) return %s;' % (Naming.nargs_cname, Naming.args_cname, code.unlikely('%s < 0' % Naming.nargs_cname), self.error_value()))
            code.putln('#endif')
            if self.signature.use_fastcall:
                code.putln('#endif')
        code.globalstate.use_utility_code(UtilityCode.load_cached('fastcall', 'FunctionArguments.c'))
        code.putln('%s = __Pyx_KwValues_%s(%s, %s);' % (Naming.kwvalues_cname, self.signature.fastvar, Naming.args_cname, Naming.nargs_cname))
        if not self.signature_has_generic_args():
            if has_star_or_kw_args:
                error(self.pos, 'This method cannot have * or keyword arguments')
            self.generate_argument_conversion_code(code)
        elif not self.signature_has_nongeneric_args():
            self.generate_stararg_copy_code(code)
        else:
            self.generate_tuple_and_keyword_parsing_code(self.args, code, decl_code)
            self.needs_values_cleanup = True
        code.error_label = old_error_label
        if code.label_used(our_error_label):
            if not code.label_used(end_label):
                code.put_goto(end_label)
            code.put_label(our_error_label)
            self.generate_argument_values_cleanup_code(code)
            if has_star_or_kw_args:
                self.generate_arg_decref(self.star_arg, code)
                if self.starstar_arg:
                    if self.starstar_arg.entry.xdecref_cleanup:
                        code.put_var_xdecref_clear(self.starstar_arg.entry)
                    else:
                        code.put_var_decref_clear(self.starstar_arg.entry)
            for arg in self.args:
                if not arg.type.is_pyobject and arg.type.needs_refcounting:
                    code.put_var_xdecref(arg.entry)
            code.put_add_traceback(self.target.entry.qualified_name)
            code.put_finish_refcount_context()
            code.putln('return %s;' % self.error_value())
        if code.label_used(end_label):
            code.put_label(end_label)

    def generate_arg_xdecref(self, arg, code):
        if arg:
            code.put_var_xdecref_clear(arg.entry)

    def generate_arg_decref(self, arg, code):
        if arg:
            code.put_var_decref_clear(arg.entry)

    def generate_stararg_copy_code(self, code):
        if not self.star_arg:
            code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseArgTupleInvalid', 'FunctionArguments.c'))
            code.putln('if (unlikely(%s > 0)) {' % Naming.nargs_cname)
            code.put('__Pyx_RaiseArgtupleInvalid(%s, 1, 0, 0, %s); return %s;' % (self.name.as_c_string_literal(), Naming.nargs_cname, self.error_value()))
            code.putln('}')
        if self.starstar_arg:
            if self.star_arg or not self.starstar_arg.entry.cf_used:
                kwarg_check = 'unlikely(%s)' % Naming.kwds_cname
            else:
                kwarg_check = '%s' % Naming.kwds_cname
        else:
            kwarg_check = 'unlikely(%s) && __Pyx_NumKwargs_%s(%s)' % (Naming.kwds_cname, self.signature.fastvar, Naming.kwds_cname)
        code.globalstate.use_utility_code(UtilityCode.load_cached('KeywordStringCheck', 'FunctionArguments.c'))
        code.putln('if (%s && unlikely(!__Pyx_CheckKeywordStrings(%s, %s, %d))) return %s;' % (kwarg_check, Naming.kwds_cname, self.name.as_c_string_literal(), bool(self.starstar_arg), self.error_value()))
        if self.starstar_arg and self.starstar_arg.entry.cf_used:
            code.putln('if (%s) {' % kwarg_check)
            code.putln('%s = __Pyx_KwargsAsDict_%s(%s, %s);' % (self.starstar_arg.entry.cname, self.signature.fastvar, Naming.kwds_cname, Naming.kwvalues_cname))
            code.putln('if (unlikely(!%s)) return %s;' % (self.starstar_arg.entry.cname, self.error_value()))
            code.put_gotref(self.starstar_arg.entry.cname, py_object_type)
            code.putln('} else {')
            code.putln('%s = PyDict_New();' % (self.starstar_arg.entry.cname,))
            code.putln('if (unlikely(!%s)) return %s;' % (self.starstar_arg.entry.cname, self.error_value()))
            code.put_var_gotref(self.starstar_arg.entry)
            self.starstar_arg.entry.xdecref_cleanup = False
            code.putln('}')
        if self.self_in_stararg and (not self.target.is_staticmethod):
            assert not self.signature.use_fastcall
            code.putln('%s = PyTuple_New(%s + 1); %s' % (self.star_arg.entry.cname, Naming.nargs_cname, code.error_goto_if_null(self.star_arg.entry.cname, self.pos)))
            code.put_var_gotref(self.star_arg.entry)
            code.put_incref(Naming.self_cname, py_object_type)
            code.put_giveref(Naming.self_cname, py_object_type)
            code.putln('PyTuple_SET_ITEM(%s, 0, %s);' % (self.star_arg.entry.cname, Naming.self_cname))
            temp = code.funcstate.allocate_temp(PyrexTypes.c_py_ssize_t_type, manage_ref=False)
            code.putln('for (%s=0; %s < %s; %s++) {' % (temp, temp, Naming.nargs_cname, temp))
            code.putln('PyObject* item = PyTuple_GET_ITEM(%s, %s);' % (Naming.args_cname, temp))
            code.put_incref('item', py_object_type)
            code.put_giveref('item', py_object_type)
            code.putln('PyTuple_SET_ITEM(%s, %s+1, item);' % (self.star_arg.entry.cname, temp))
            code.putln('}')
            code.funcstate.release_temp(temp)
            self.star_arg.entry.xdecref_cleanup = 0
        elif self.star_arg:
            assert not self.signature.use_fastcall
            code.put_incref(Naming.args_cname, py_object_type)
            code.putln('%s = %s;' % (self.star_arg.entry.cname, Naming.args_cname))
            self.star_arg.entry.xdecref_cleanup = 0

    def generate_tuple_and_keyword_parsing_code(self, args, code, decl_code):
        code.globalstate.use_utility_code(UtilityCode.load_cached('fastcall', 'FunctionArguments.c'))
        self_name_csafe = self.name.as_c_string_literal()
        argtuple_error_label = code.new_label('argtuple_error')
        positional_args = []
        required_kw_only_args = []
        optional_kw_only_args = []
        num_pos_only_args = 0
        for arg in args:
            if arg.is_generic:
                if arg.default:
                    if not arg.is_self_arg and (not arg.is_type_arg):
                        if arg.kw_only:
                            optional_kw_only_args.append(arg)
                        else:
                            positional_args.append(arg)
                elif arg.kw_only:
                    required_kw_only_args.append(arg)
                elif not arg.is_self_arg and (not arg.is_type_arg):
                    positional_args.append(arg)
                if arg.pos_only:
                    num_pos_only_args += 1
        kw_only_args = required_kw_only_args + optional_kw_only_args
        min_positional_args = self.num_required_args - self.num_required_kw_args
        if len(args) > 0 and (args[0].is_self_arg or args[0].is_type_arg):
            min_positional_args -= 1
        max_positional_args = len(positional_args)
        has_fixed_positional_count = not self.star_arg and min_positional_args == max_positional_args
        has_kw_only_args = bool(kw_only_args)
        if self.starstar_arg or self.star_arg:
            self.generate_stararg_init_code(max_positional_args, code)
        code.putln('{')
        all_args = tuple(positional_args) + tuple(kw_only_args)
        non_posonly_args = [arg for arg in all_args if not arg.pos_only]
        non_pos_args_id = ','.join(['&%s' % code.intern_identifier(arg.entry.name) for arg in non_posonly_args] + ['0'])
        code.putln('PyObject **%s[] = {%s};' % (Naming.pykwdlist_cname, non_pos_args_id))
        self.generate_argument_values_setup_code(all_args, code, decl_code)
        accept_kwd_args = non_posonly_args or self.starstar_arg
        if accept_kwd_args:
            kw_unpacking_condition = Naming.kwds_cname
        else:
            kw_unpacking_condition = '%s && __Pyx_NumKwargs_%s(%s) > 0' % (Naming.kwds_cname, self.signature.fastvar, Naming.kwds_cname)
        if self.num_required_kw_args > 0:
            kw_unpacking_condition = 'likely(%s)' % kw_unpacking_condition
        code.putln('if (%s) {' % kw_unpacking_condition)
        if accept_kwd_args:
            self.generate_keyword_unpacking_code(min_positional_args, max_positional_args, has_fixed_positional_count, has_kw_only_args, all_args, argtuple_error_label, code)
        else:
            code.globalstate.use_utility_code(UtilityCode.load_cached('ParseKeywords', 'FunctionArguments.c'))
            code.putln('if (likely(__Pyx_ParseOptionalKeywords(%s, %s, %s, %s, %s, %s, %s) < 0)) %s' % (Naming.kwds_cname, Naming.kwvalues_cname, Naming.pykwdlist_cname, self.starstar_arg.entry.cname if self.starstar_arg else 0, 'values', 0, self_name_csafe, code.error_goto(self.pos)))
        if self.num_required_kw_args and min_positional_args > 0 or min_positional_args == max_positional_args:
            if min_positional_args == max_positional_args and (not self.star_arg):
                compare = '!='
            else:
                compare = '<'
            code.putln('} else if (unlikely(%s %s %d)) {' % (Naming.nargs_cname, compare, min_positional_args))
            code.put_goto(argtuple_error_label)
        if self.num_required_kw_args:
            if max_positional_args > min_positional_args and (not self.star_arg):
                code.putln('} else if (unlikely(%s > %d)) {' % (Naming.nargs_cname, max_positional_args))
                code.put_goto(argtuple_error_label)
            code.putln('} else {')
            for i, arg in enumerate(kw_only_args):
                if not arg.default:
                    pystring_cname = code.intern_identifier(arg.entry.name)
                    code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseKeywordRequired', 'FunctionArguments.c'))
                    code.put('__Pyx_RaiseKeywordRequired("%s", %s); ' % (self.name, pystring_cname))
                    code.putln(code.error_goto(self.pos))
                    break
        else:
            code.putln('} else {')
            if min_positional_args == max_positional_args:
                for i, arg in enumerate(positional_args):
                    code.putln('values[%d] = __Pyx_Arg_%s(%s, %d);' % (i, self.signature.fastvar, Naming.args_cname, i))
            else:
                code.putln('switch (%s) {' % Naming.nargs_cname)
                if self.star_arg:
                    code.putln('default:')
                reversed_args = list(enumerate(positional_args))[::-1]
                for i, arg in reversed_args:
                    if i >= min_positional_args - 1:
                        if i != reversed_args[0][0]:
                            code.putln('CYTHON_FALLTHROUGH;')
                        code.put('case %2d: ' % (i + 1))
                    code.putln('values[%d] = __Pyx_Arg_%s(%s, %d);' % (i, self.signature.fastvar, Naming.args_cname, i))
                if min_positional_args == 0:
                    code.putln('CYTHON_FALLTHROUGH;')
                    code.put('case  0: ')
                code.putln('break;')
                if self.star_arg:
                    if min_positional_args:
                        for i in range(min_positional_args - 1, -1, -1):
                            code.putln('case %2d:' % i)
                        code.put_goto(argtuple_error_label)
                else:
                    code.put('default: ')
                    code.put_goto(argtuple_error_label)
                code.putln('}')
        code.putln('}')
        for i, arg in enumerate(all_args):
            self.generate_arg_assignment(arg, 'values[%d]' % i, code)
        code.putln('}')
        if code.label_used(argtuple_error_label):
            skip_error_handling = code.new_label('skip')
            code.put_goto(skip_error_handling)
            code.put_label(argtuple_error_label)
            code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseArgTupleInvalid', 'FunctionArguments.c'))
            code.putln('__Pyx_RaiseArgtupleInvalid(%s, %d, %d, %d, %s); %s' % (self_name_csafe, has_fixed_positional_count, min_positional_args, max_positional_args, Naming.nargs_cname, code.error_goto(self.pos)))
            code.put_label(skip_error_handling)

    def generate_arg_assignment(self, arg, item, code):
        if arg.type.is_pyobject:
            if arg.is_generic:
                item = PyrexTypes.typecast(arg.type, PyrexTypes.py_object_type, item)
            entry = arg.entry
            code.putln('%s = %s;' % (entry.cname, item))
        elif arg.type.from_py_function:
            if arg.default:
                code.putln('if (%s) {' % item)
            code.putln(arg.type.from_py_call_code(item, arg.entry.cname, arg.pos, code))
            if arg.default:
                code.putln('} else {')
                code.putln('%s = %s;' % (arg.entry.cname, arg.calculate_default_value_code(code)))
                if arg.type.is_memoryviewslice:
                    code.put_var_incref_memoryviewslice(arg.entry, have_gil=True)
                code.putln('}')
        else:
            error(arg.pos, "Cannot convert Python object argument to type '%s'" % arg.type)

    def generate_stararg_init_code(self, max_positional_args, code):
        if self.starstar_arg:
            self.starstar_arg.entry.xdecref_cleanup = 0
            code.putln('%s = PyDict_New(); if (unlikely(!%s)) return %s;' % (self.starstar_arg.entry.cname, self.starstar_arg.entry.cname, self.error_value()))
            code.put_var_gotref(self.starstar_arg.entry)
        if self.star_arg:
            self.star_arg.entry.xdecref_cleanup = 0
            if max_positional_args == 0:
                assert not self.signature.use_fastcall
                code.put_incref(Naming.args_cname, py_object_type)
                code.putln('%s = %s;' % (self.star_arg.entry.cname, Naming.args_cname))
            else:
                code.putln('%s = __Pyx_ArgsSlice_%s(%s, %d, %s);' % (self.star_arg.entry.cname, self.signature.fastvar, Naming.args_cname, max_positional_args, Naming.nargs_cname))
                code.putln('if (unlikely(!%s)) {' % self.star_arg.entry.type.nullcheck_string(self.star_arg.entry.cname))
                if self.starstar_arg:
                    code.put_var_decref_clear(self.starstar_arg.entry)
                code.put_finish_refcount_context()
                code.putln('return %s;' % self.error_value())
                code.putln('}')
                code.put_var_gotref(self.star_arg.entry)

    def generate_argument_values_setup_code(self, args, code, decl_code):
        max_args = len(args)
        decl_code.putln('PyObject* values[%d] = {%s};' % (max_args, ','.join('0' * max_args)))
        if self.target.defaults_struct:
            code.putln('%s *%s = __Pyx_CyFunction_Defaults(%s, %s);' % (self.target.defaults_struct, Naming.dynamic_args_cname, self.target.defaults_struct, Naming.self_cname))
        for i, arg in enumerate(args):
            if arg.default and arg.type.is_pyobject:
                default_value = arg.calculate_default_value_code(code)
                code.putln('values[%d] = __Pyx_Arg_NewRef_%s(%s);' % (i, self.signature.fastvar, arg.type.as_pyobject(default_value)))

    def generate_argument_values_cleanup_code(self, code):
        if not self.needs_values_cleanup:
            return
        loop_var = Naming.quick_temp_cname
        code.putln('{')
        code.putln('Py_ssize_t %s;' % loop_var)
        code.putln('for (%s=0; %s < (Py_ssize_t)(sizeof(values)/sizeof(values[0])); ++%s) {' % (loop_var, loop_var, loop_var))
        code.putln('__Pyx_Arg_XDECREF_%s(values[%s]);' % (self.signature.fastvar, loop_var))
        code.putln('}')
        code.putln('}')

    def generate_keyword_unpacking_code(self, min_positional_args, max_positional_args, has_fixed_positional_count, has_kw_only_args, all_args, argtuple_error_label, code):
        num_required_posonly_args = num_pos_only_args = 0
        for i, arg in enumerate(all_args):
            if arg.pos_only:
                num_pos_only_args += 1
                if not arg.default:
                    num_required_posonly_args += 1
        code.putln('Py_ssize_t kw_args;')
        code.putln('switch (%s) {' % Naming.nargs_cname)
        if self.star_arg:
            code.putln('default:')
        for i in range(max_positional_args - 1, num_required_posonly_args - 1, -1):
            code.put('case %2d: ' % (i + 1))
            code.putln('values[%d] = __Pyx_Arg_%s(%s, %d);' % (i, self.signature.fastvar, Naming.args_cname, i))
            code.putln('CYTHON_FALLTHROUGH;')
        if num_required_posonly_args > 0:
            code.put('case %2d: ' % num_required_posonly_args)
            for i in range(num_required_posonly_args - 1, -1, -1):
                code.putln('values[%d] = __Pyx_Arg_%s(%s, %d);' % (i, self.signature.fastvar, Naming.args_cname, i))
            code.putln('break;')
        for i in range(num_required_posonly_args - 2, -1, -1):
            code.put('case %2d: ' % (i + 1))
            code.putln('CYTHON_FALLTHROUGH;')
        code.put('case  0: ')
        if num_required_posonly_args == 0:
            code.putln('break;')
        else:
            code.put_goto(argtuple_error_label)
        if not self.star_arg:
            code.put('default: ')
            code.put_goto(argtuple_error_label)
        code.putln('}')
        self_name_csafe = self.name.as_c_string_literal()
        code.putln('kw_args = __Pyx_NumKwargs_%s(%s);' % (self.signature.fastvar, Naming.kwds_cname))
        if self.num_required_args or max_positional_args > 0:
            last_required_arg = -1
            for i, arg in enumerate(all_args):
                if not arg.default:
                    last_required_arg = i
            if last_required_arg < max_positional_args:
                last_required_arg = max_positional_args - 1
            if max_positional_args > num_pos_only_args:
                code.putln('switch (%s) {' % Naming.nargs_cname)
            for i, arg in enumerate(all_args[num_pos_only_args:last_required_arg + 1], num_pos_only_args):
                if max_positional_args > num_pos_only_args and i <= max_positional_args:
                    if i != num_pos_only_args:
                        code.putln('CYTHON_FALLTHROUGH;')
                    if self.star_arg and i == max_positional_args:
                        code.putln('default:')
                    else:
                        code.putln('case %2d:' % i)
                pystring_cname = code.intern_identifier(arg.entry.name)
                if arg.default:
                    if arg.kw_only:
                        continue
                    code.putln('if (kw_args > 0) {')
                    code.putln('PyObject* value = __Pyx_GetKwValue_%s(%s, %s, %s);' % (self.signature.fastvar, Naming.kwds_cname, Naming.kwvalues_cname, pystring_cname))
                    code.putln('if (value) { values[%d] = __Pyx_Arg_NewRef_%s(value); kw_args--; }' % (i, self.signature.fastvar))
                    code.putln('else if (unlikely(PyErr_Occurred())) %s' % code.error_goto(self.pos))
                    code.putln('}')
                else:
                    code.putln('if (likely((values[%d] = __Pyx_GetKwValue_%s(%s, %s, %s)) != 0)) {' % (i, self.signature.fastvar, Naming.kwds_cname, Naming.kwvalues_cname, pystring_cname))
                    code.putln('(void)__Pyx_Arg_NewRef_%s(values[%d]);' % (self.signature.fastvar, i))
                    code.putln('kw_args--;')
                    code.putln('}')
                    code.putln('else if (unlikely(PyErr_Occurred())) %s' % code.error_goto(self.pos))
                    if i < min_positional_args:
                        if i == 0:
                            code.put('else ')
                            code.put_goto(argtuple_error_label)
                        else:
                            code.putln('else {')
                            code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseArgTupleInvalid', 'FunctionArguments.c'))
                            code.put('__Pyx_RaiseArgtupleInvalid(%s, %d, %d, %d, %d); ' % (self_name_csafe, has_fixed_positional_count, min_positional_args, max_positional_args, i))
                            code.putln(code.error_goto(self.pos))
                            code.putln('}')
                    elif arg.kw_only:
                        code.putln('else {')
                        code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseKeywordRequired', 'FunctionArguments.c'))
                        code.put('__Pyx_RaiseKeywordRequired(%s, %s); ' % (self_name_csafe, pystring_cname))
                        code.putln(code.error_goto(self.pos))
                        code.putln('}')
            if max_positional_args > num_pos_only_args:
                code.putln('}')
        if has_kw_only_args:
            self.generate_optional_kwonly_args_unpacking_code(all_args, code)
        code.putln('if (unlikely(kw_args > 0)) {')
        if num_pos_only_args > 0:
            code.putln('const Py_ssize_t kwd_pos_args = (unlikely(%s < %d)) ? 0 : %s - %d;' % (Naming.nargs_cname, num_pos_only_args, Naming.nargs_cname, num_pos_only_args))
        elif max_positional_args > 0:
            code.putln('const Py_ssize_t kwd_pos_args = %s;' % Naming.nargs_cname)
        if max_positional_args == 0:
            pos_arg_count = '0'
        elif self.star_arg:
            code.putln('const Py_ssize_t used_pos_args = (kwd_pos_args < %d) ? kwd_pos_args : %d;' % (max_positional_args - num_pos_only_args, max_positional_args - num_pos_only_args))
            pos_arg_count = 'used_pos_args'
        else:
            pos_arg_count = 'kwd_pos_args'
        if num_pos_only_args < len(all_args):
            values_array = 'values + %d' % num_pos_only_args
        else:
            values_array = 'values'
        code.globalstate.use_utility_code(UtilityCode.load_cached('ParseKeywords', 'FunctionArguments.c'))
        code.putln('if (unlikely(__Pyx_ParseOptionalKeywords(%s, %s, %s, %s, %s, %s, %s) < 0)) %s' % (Naming.kwds_cname, Naming.kwvalues_cname, Naming.pykwdlist_cname, self.starstar_arg and self.starstar_arg.entry.cname or '0', values_array, pos_arg_count, self_name_csafe, code.error_goto(self.pos)))
        code.putln('}')

    def generate_optional_kwonly_args_unpacking_code(self, all_args, code):
        optional_args = []
        first_optional_arg = -1
        num_posonly_args = 0
        for i, arg in enumerate(all_args):
            if arg.pos_only:
                num_posonly_args += 1
            if not arg.kw_only or not arg.default:
                continue
            if not optional_args:
                first_optional_arg = i
            optional_args.append(arg.name)
        if num_posonly_args > 0:
            posonly_correction = '-%d' % num_posonly_args
        else:
            posonly_correction = ''
        if optional_args:
            if len(optional_args) > 1:
                code.putln('if (kw_args > 0 && %s(kw_args <= %d)) {' % (not self.starstar_arg and 'likely' or '', len(optional_args)))
                code.putln('Py_ssize_t index;')
                code.putln('for (index = %d; index < %d && kw_args > 0; index++) {' % (first_optional_arg, first_optional_arg + len(optional_args)))
            else:
                code.putln('if (kw_args == 1) {')
                code.putln('const Py_ssize_t index = %d;' % first_optional_arg)
            code.putln('PyObject* value = __Pyx_GetKwValue_%s(%s, %s, *%s[index%s]);' % (self.signature.fastvar, Naming.kwds_cname, Naming.kwvalues_cname, Naming.pykwdlist_cname, posonly_correction))
            code.putln('if (value) { values[index] = __Pyx_Arg_NewRef_%s(value); kw_args--; }' % self.signature.fastvar)
            code.putln('else if (unlikely(PyErr_Occurred())) %s' % code.error_goto(self.pos))
            if len(optional_args) > 1:
                code.putln('}')
            code.putln('}')

    def generate_argument_conversion_code(self, code):
        for arg in self.args:
            if arg.needs_conversion:
                self.generate_arg_conversion(arg, code)

    def generate_arg_conversion(self, arg, code):
        old_type = arg.hdr_type
        new_type = arg.type
        if old_type.is_pyobject:
            if arg.default:
                code.putln('if (%s) {' % arg.hdr_cname)
            else:
                code.putln('assert(%s); {' % arg.hdr_cname)
            self.generate_arg_conversion_from_pyobject(arg, code)
            code.putln('}')
        elif new_type.is_pyobject:
            self.generate_arg_conversion_to_pyobject(arg, code)
        elif new_type.assignable_from(old_type):
            code.putln('%s = %s;' % (arg.entry.cname, arg.hdr_cname))
        else:
            error(arg.pos, "Cannot convert 1 argument from '%s' to '%s'" % (old_type, new_type))

    def generate_arg_conversion_from_pyobject(self, arg, code):
        new_type = arg.type
        if new_type.from_py_function:
            code.putln(new_type.from_py_call_code(arg.hdr_cname, arg.entry.cname, arg.pos, code))
        else:
            error(arg.pos, "Cannot convert Python object argument to type '%s'" % new_type)

    def generate_arg_conversion_to_pyobject(self, arg, code):
        old_type = arg.hdr_type
        func = old_type.to_py_function
        if func:
            code.putln('%s = %s(%s); %s' % (arg.entry.cname, func, arg.hdr_cname, code.error_goto_if_null(arg.entry.cname, arg.pos)))
            code.put_var_gotref(arg.entry)
        else:
            error(arg.pos, "Cannot convert argument of type '%s' to Python object" % old_type)

    def generate_argument_type_tests(self, code):
        for arg in self.args:
            if arg.needs_type_test:
                self.generate_arg_type_test(arg, code)
            elif not arg.accept_none and (arg.type.is_pyobject or arg.type.is_buffer or arg.type.is_memoryviewslice):
                self.generate_arg_none_check(arg, code)
        if self.target.entry.is_special:
            for n in reversed(range(len(self.args), self.signature.max_num_fixed_args())):
                if self.target.entry.name == '__ipow__':
                    code.putln('#if PY_VERSION_HEX >= 0x03080000')
                code.putln('if (unlikely(unused_arg_%s != Py_None)) {' % n)
                code.putln('PyErr_SetString(PyExc_TypeError, "%s() takes %s arguments but %s were given");' % (self.target.entry.qualified_name, self.signature.max_num_fixed_args(), n))
                code.putln('%s;' % code.error_goto(self.pos))
                code.putln('}')
                if self.target.entry.name == '__ipow__':
                    code.putln('#endif /*PY_VERSION_HEX >= 0x03080000*/')
            if self.target.entry.name == '__ipow__' and len(self.args) != 2:
                code.putln('if ((PY_VERSION_HEX < 0x03080000)) {')
                code.putln('PyErr_SetString(PyExc_NotImplementedError, "3-argument %s cannot be used in Python<3.8");' % self.target.entry.qualified_name)
                code.putln('%s;' % code.error_goto(self.pos))
                code.putln('}')

    def error_value(self):
        return self.signature.error_value