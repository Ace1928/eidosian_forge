from __future__ import absolute_import
import cython
import hashlib
import operator
import os
import re
import shutil
import textwrap
from string import Template
from functools import partial
from contextlib import closing, contextmanager
from collections import defaultdict
from . import Naming
from . import Options
from . import DebugFlags
from . import StringEncoding
from .. import Utils
from .Scanning import SourceDescriptor
from ..StringIOTree import StringIOTree
class CCodeWriter(object):
    """
    Utility class to output C code.

    When creating an insertion point one must care about the state that is
    kept:
    - formatting state (level, bol) is cloned and used in insertion points
      as well
    - labels, temps, exc_vars: One must construct a scope in which these can
      exist by calling enter_cfunc_scope/exit_cfunc_scope (these are for
      sanity checking and forward compatibility). Created insertion points
      looses this scope and cannot access it.
    - marker: Not copied to insertion point
    - filename_table, filename_list, input_file_contents: All codewriters
      coming from the same root share the same instances simultaneously.
    """

    @cython.locals(create_from='CCodeWriter')
    def __init__(self, create_from=None, buffer=None, copy_formatting=False):
        if buffer is None:
            buffer = StringIOTree()
        self.buffer = buffer
        self.last_pos = None
        self.last_marked_pos = None
        self.pyclass_stack = []
        self.funcstate = None
        self.globalstate = None
        self.code_config = None
        self.level = 0
        self.call_level = 0
        self.bol = 1
        if create_from is not None:
            self.set_global_state(create_from.globalstate)
            self.funcstate = create_from.funcstate
            if copy_formatting:
                self.level = create_from.level
                self.bol = create_from.bol
                self.call_level = create_from.call_level
            self.last_pos = create_from.last_pos
            self.last_marked_pos = create_from.last_marked_pos

    def create_new(self, create_from, buffer, copy_formatting):
        result = CCodeWriter(create_from, buffer, copy_formatting)
        return result

    def set_global_state(self, global_state):
        assert self.globalstate is None
        self.globalstate = global_state
        self.code_config = global_state.code_config

    def copyto(self, f):
        self.buffer.copyto(f)

    def getvalue(self):
        return self.buffer.getvalue()

    def write(self, s):
        if '\n' in s:
            self._write_lines(s)
        else:
            self._write_to_buffer(s)

    def _write_lines(self, s):
        filename_line = self.last_marked_pos[:2] if self.last_marked_pos else (None, 0)
        self.buffer.markers.extend([filename_line] * s.count('\n'))
        self._write_to_buffer(s)

    def _write_to_buffer(self, s):
        self.buffer.write(s)

    def insertion_point(self):
        other = self.create_new(create_from=self, buffer=self.buffer.insertion_point(), copy_formatting=True)
        return other

    def new_writer(self):
        """
        Creates a new CCodeWriter connected to the same global state, which
        can later be inserted using insert.
        """
        return CCodeWriter(create_from=self)

    def insert(self, writer):
        """
        Inserts the contents of another code writer (created with
        the same global state) in the current location.

        It is ok to write to the inserted writer also after insertion.
        """
        assert writer.globalstate is self.globalstate
        self.buffer.insert(writer.buffer)

    @funccontext_property
    def label_counter(self):
        pass

    @funccontext_property
    def return_label(self):
        pass

    @funccontext_property
    def error_label(self):
        pass

    @funccontext_property
    def labels_used(self):
        pass

    @funccontext_property
    def continue_label(self):
        pass

    @funccontext_property
    def break_label(self):
        pass

    @funccontext_property
    def return_from_error_cleanup_label(self):
        pass

    @funccontext_property
    def yield_labels(self):
        pass

    def label_interceptor(self, new_labels, orig_labels, skip_to_label=None, pos=None, trace=True):
        """
        Helper for generating multiple label interceptor code blocks.

        @param new_labels: the new labels that should be intercepted
        @param orig_labels: the original labels that we should dispatch to after the interception
        @param skip_to_label: a label to skip to before starting the code blocks
        @param pos: the node position to mark for each interceptor block
        @param trace: add a trace line for the pos marker or not
        """
        for label, orig_label in zip(new_labels, orig_labels):
            if not self.label_used(label):
                continue
            if skip_to_label:
                self.put_goto(skip_to_label)
                skip_to_label = None
            if pos is not None:
                self.mark_pos(pos, trace=trace)
            self.put_label(label)
            yield (label, orig_label)
            self.put_goto(orig_label)

    def new_label(self, name=None):
        return self.funcstate.new_label(name)

    def new_error_label(self, *args):
        return self.funcstate.new_error_label(*args)

    def new_yield_label(self, *args):
        return self.funcstate.new_yield_label(*args)

    def get_loop_labels(self):
        return self.funcstate.get_loop_labels()

    def set_loop_labels(self, labels):
        return self.funcstate.set_loop_labels(labels)

    def new_loop_labels(self, *args):
        return self.funcstate.new_loop_labels(*args)

    def get_all_labels(self):
        return self.funcstate.get_all_labels()

    def set_all_labels(self, labels):
        return self.funcstate.set_all_labels(labels)

    def all_new_labels(self):
        return self.funcstate.all_new_labels()

    def use_label(self, lbl):
        return self.funcstate.use_label(lbl)

    def label_used(self, lbl):
        return self.funcstate.label_used(lbl)

    def enter_cfunc_scope(self, scope=None):
        self.funcstate = FunctionState(self, scope=scope)

    def exit_cfunc_scope(self):
        self.funcstate.validate_exit()
        self.funcstate = None

    def get_py_int(self, str_value, longness):
        return self.globalstate.get_int_const(str_value, longness).cname

    def get_py_float(self, str_value, value_code):
        return self.globalstate.get_float_const(str_value, value_code).cname

    def get_py_const(self, type, prefix='', cleanup_level=None, dedup_key=None):
        return self.globalstate.get_py_const(type, prefix, cleanup_level, dedup_key).cname

    def get_string_const(self, text):
        return self.globalstate.get_string_const(text).cname

    def get_pyunicode_ptr_const(self, text):
        return self.globalstate.get_pyunicode_ptr_const(text)

    def get_py_string_const(self, text, identifier=None, is_str=False, unicode_value=None):
        return self.globalstate.get_py_string_const(text, identifier, is_str, unicode_value).cname

    def get_argument_default_const(self, type):
        return self.globalstate.get_py_const(type).cname

    def intern(self, text):
        return self.get_py_string_const(text)

    def intern_identifier(self, text):
        return self.get_py_string_const(text, identifier=True)

    def get_cached_constants_writer(self, target=None):
        return self.globalstate.get_cached_constants_writer(target)

    def putln(self, code='', safe=False):
        if self.last_pos and self.bol:
            self.emit_marker()
        if self.code_config.emit_linenums and self.last_marked_pos:
            source_desc, line, _ = self.last_marked_pos
            self._write_lines('\n#line %s "%s"\n' % (line, source_desc.get_escaped_description()))
        if code:
            if safe:
                self.put_safe(code)
            else:
                self.put(code)
        self._write_lines('\n')
        self.bol = 1

    def mark_pos(self, pos, trace=True):
        if pos is None:
            return
        if self.last_marked_pos and self.last_marked_pos[:2] == pos[:2]:
            return
        self.last_pos = (pos, trace)

    def emit_marker(self):
        pos, trace = self.last_pos
        self.last_marked_pos = pos
        self.last_pos = None
        self._write_lines('\n')
        if self.code_config.emit_code_comments:
            self.indent()
            self._write_lines('/* %s */\n' % self._build_marker(pos))
        if trace and self.funcstate and self.funcstate.can_trace and self.globalstate.directives['linetrace']:
            self.indent()
            self._write_lines('__Pyx_TraceLine(%d,%d,%s)\n' % (pos[1], not self.funcstate.gil_owned, self.error_goto(pos)))

    def _build_marker(self, pos):
        source_desc, line, col = pos
        assert isinstance(source_desc, SourceDescriptor)
        contents = self.globalstate.commented_file_contents(source_desc)
        lines = contents[max(0, line - 3):line]
        lines[-1] += u'             # <<<<<<<<<<<<<<'
        lines += contents[line:line + 2]
        return u'"%s":%d\n%s\n' % (source_desc.get_escaped_description(), line, u'\n'.join(lines))

    def put_safe(self, code):
        self.write(code)
        self.bol = 0

    def put_or_include(self, code, name):
        include_dir = self.globalstate.common_utility_include_dir
        if include_dir and len(code) > 1024:
            include_file = '%s_%s.h' % (name, hashlib.sha1(code.encode('utf8')).hexdigest())
            path = os.path.join(include_dir, include_file)
            if not os.path.exists(path):
                tmp_path = '%s.tmp%s' % (path, os.getpid())
                with closing(Utils.open_new_file(tmp_path)) as f:
                    f.write(code)
                shutil.move(tmp_path, path)
            code = '#include "%s"\n' % path
        self.put(code)

    def put(self, code):
        fix_indent = False
        if '{' in code:
            dl = code.count('{')
        else:
            dl = 0
        if '}' in code:
            dl -= code.count('}')
            if dl < 0:
                self.level += dl
            elif dl == 0 and code[0] == '}':
                fix_indent = True
                self.level -= 1
        if self.bol:
            self.indent()
        self.write(code)
        self.bol = 0
        if dl > 0:
            self.level += dl
        elif fix_indent:
            self.level += 1

    def putln_tempita(self, code, **context):
        from ..Tempita import sub
        self.putln(sub(code, **context))

    def put_tempita(self, code, **context):
        from ..Tempita import sub
        self.put(sub(code, **context))

    def increase_indent(self):
        self.level += 1

    def decrease_indent(self):
        self.level -= 1

    def begin_block(self):
        self.putln('{')
        self.increase_indent()

    def end_block(self):
        self.decrease_indent()
        self.putln('}')

    def indent(self):
        self._write_to_buffer('  ' * self.level)

    def get_py_version_hex(self, pyversion):
        return '0x%02X%02X%02X%02X' % (tuple(pyversion) + (0, 0, 0, 0))[:4]

    def put_label(self, lbl):
        if lbl in self.funcstate.labels_used:
            self.putln('%s:;' % lbl)

    def put_goto(self, lbl):
        self.funcstate.use_label(lbl)
        self.putln('goto %s;' % lbl)

    def put_var_declaration(self, entry, storage_class='', dll_linkage=None, definition=True):
        if entry.visibility == 'private' and (not (definition or entry.defined_in_pxd)):
            return
        if entry.visibility == 'private' and (not entry.used):
            return
        if not entry.cf_used:
            self.put('CYTHON_UNUSED ')
        if storage_class:
            self.put('%s ' % storage_class)
        if entry.is_cpp_optional:
            self.put(entry.type.cpp_optional_declaration_code(entry.cname, dll_linkage=dll_linkage))
        else:
            self.put(entry.type.declaration_code(entry.cname, dll_linkage=dll_linkage))
        if entry.init is not None:
            self.put_safe(' = %s' % entry.type.literal_code(entry.init))
        elif entry.type.is_pyobject:
            self.put(' = NULL')
        self.putln(';')
        self.funcstate.scope.use_entry_utility_code(entry)

    def put_temp_declarations(self, func_context):
        for name, type, manage_ref, static in func_context.temps_allocated:
            if type.is_cpp_class and (not type.is_fake_reference) and func_context.scope.directives['cpp_locals']:
                decl = type.cpp_optional_declaration_code(name)
            else:
                decl = type.declaration_code(name)
            if type.is_pyobject:
                self.putln('%s = NULL;' % decl)
            elif type.is_memoryviewslice:
                self.putln('%s = %s;' % (decl, type.literal_code(type.default_value)))
            else:
                self.putln('%s%s;' % (static and 'static ' or '', decl))
        if func_context.should_declare_error_indicator:
            if self.funcstate.uses_error_indicator:
                unused = ''
            else:
                unused = 'CYTHON_UNUSED '
            self.putln('%sint %s = 0;' % (unused, Naming.lineno_cname))
            self.putln('%sconst char *%s = NULL;' % (unused, Naming.filename_cname))
            self.putln('%sint %s = 0;' % (unused, Naming.clineno_cname))

    def put_generated_by(self):
        self.putln(Utils.GENERATED_BY_MARKER)
        self.putln('')

    def put_h_guard(self, guard):
        self.putln('#ifndef %s' % guard)
        self.putln('#define %s' % guard)

    def unlikely(self, cond):
        if Options.gcc_branch_hints:
            return 'unlikely(%s)' % cond
        else:
            return cond

    def build_function_modifiers(self, modifiers, mapper=modifier_output_mapper):
        if not modifiers:
            return ''
        return '%s ' % ' '.join([mapper(m, m) for m in modifiers])

    def entry_as_pyobject(self, entry):
        type = entry.type
        if not entry.is_self_arg and (not entry.type.is_complete()) or entry.type.is_extension_type:
            return '(PyObject *)' + entry.cname
        else:
            return entry.cname

    def as_pyobject(self, cname, type):
        from .PyrexTypes import py_object_type, typecast
        return typecast(py_object_type, type, cname)

    def put_gotref(self, cname, type):
        type.generate_gotref(self, cname)

    def put_giveref(self, cname, type):
        type.generate_giveref(self, cname)

    def put_xgiveref(self, cname, type):
        type.generate_xgiveref(self, cname)

    def put_xgotref(self, cname, type):
        type.generate_xgotref(self, cname)

    def put_incref(self, cname, type, nanny=True):
        type.generate_incref(self, cname, nanny=nanny)

    def put_xincref(self, cname, type, nanny=True):
        type.generate_xincref(self, cname, nanny=nanny)

    def put_decref(self, cname, type, nanny=True, have_gil=True):
        type.generate_decref(self, cname, nanny=nanny, have_gil=have_gil)

    def put_xdecref(self, cname, type, nanny=True, have_gil=True):
        type.generate_xdecref(self, cname, nanny=nanny, have_gil=have_gil)

    def put_decref_clear(self, cname, type, clear_before_decref=False, nanny=True, have_gil=True):
        type.generate_decref_clear(self, cname, clear_before_decref=clear_before_decref, nanny=nanny, have_gil=have_gil)

    def put_xdecref_clear(self, cname, type, clear_before_decref=False, nanny=True, have_gil=True):
        type.generate_xdecref_clear(self, cname, clear_before_decref=clear_before_decref, nanny=nanny, have_gil=have_gil)

    def put_decref_set(self, cname, type, rhs_cname):
        type.generate_decref_set(self, cname, rhs_cname)

    def put_xdecref_set(self, cname, type, rhs_cname):
        type.generate_xdecref_set(self, cname, rhs_cname)

    def put_incref_memoryviewslice(self, slice_cname, type, have_gil):
        type.generate_incref_memoryviewslice(self, slice_cname, have_gil=have_gil)

    def put_var_incref_memoryviewslice(self, entry, have_gil):
        self.put_incref_memoryviewslice(entry.cname, entry.type, have_gil=have_gil)

    def put_var_gotref(self, entry):
        self.put_gotref(entry.cname, entry.type)

    def put_var_giveref(self, entry):
        self.put_giveref(entry.cname, entry.type)

    def put_var_xgotref(self, entry):
        self.put_xgotref(entry.cname, entry.type)

    def put_var_xgiveref(self, entry):
        self.put_xgiveref(entry.cname, entry.type)

    def put_var_incref(self, entry, **kwds):
        self.put_incref(entry.cname, entry.type, **kwds)

    def put_var_xincref(self, entry, **kwds):
        self.put_xincref(entry.cname, entry.type, **kwds)

    def put_var_decref(self, entry, **kwds):
        self.put_decref(entry.cname, entry.type, **kwds)

    def put_var_xdecref(self, entry, **kwds):
        self.put_xdecref(entry.cname, entry.type, **kwds)

    def put_var_decref_clear(self, entry, **kwds):
        self.put_decref_clear(entry.cname, entry.type, clear_before_decref=entry.in_closure, **kwds)

    def put_var_decref_set(self, entry, rhs_cname, **kwds):
        self.put_decref_set(entry.cname, entry.type, rhs_cname, **kwds)

    def put_var_xdecref_set(self, entry, rhs_cname, **kwds):
        self.put_xdecref_set(entry.cname, entry.type, rhs_cname, **kwds)

    def put_var_xdecref_clear(self, entry, **kwds):
        self.put_xdecref_clear(entry.cname, entry.type, clear_before_decref=entry.in_closure, **kwds)

    def put_var_decrefs(self, entries, used_only=0):
        for entry in entries:
            if not used_only or entry.used:
                if entry.xdecref_cleanup:
                    self.put_var_xdecref(entry)
                else:
                    self.put_var_decref(entry)

    def put_var_xdecrefs(self, entries):
        for entry in entries:
            self.put_var_xdecref(entry)

    def put_var_xdecrefs_clear(self, entries):
        for entry in entries:
            self.put_var_xdecref_clear(entry)

    def put_init_to_py_none(self, cname, type, nanny=True):
        from .PyrexTypes import py_object_type, typecast
        py_none = typecast(type, py_object_type, 'Py_None')
        if nanny:
            self.putln('%s = %s; __Pyx_INCREF(Py_None);' % (cname, py_none))
        else:
            self.putln('%s = %s; Py_INCREF(Py_None);' % (cname, py_none))

    def put_init_var_to_py_none(self, entry, template='%s', nanny=True):
        code = template % entry.cname
        self.put_init_to_py_none(code, entry.type, nanny)
        if entry.in_closure:
            self.put_giveref('Py_None')

    def put_pymethoddef(self, entry, term, allow_skip=True, wrapper_code_writer=None):
        is_reverse_number_slot = False
        if entry.is_special or entry.name == '__getattribute__':
            from . import TypeSlots
            is_reverse_number_slot = True
            if entry.name not in special_py_methods and (not TypeSlots.is_reverse_number_slot(entry.name)):
                if entry.name == '__getattr__' and (not self.globalstate.directives['fast_getattr']):
                    pass
                elif allow_skip:
                    return
        method_flags = entry.signature.method_flags()
        if not method_flags:
            return
        if entry.is_special:
            method_flags += [TypeSlots.method_coexist]
        func_ptr = wrapper_code_writer.put_pymethoddef_wrapper(entry) if wrapper_code_writer else entry.func_cname
        cast = entry.signature.method_function_type()
        if cast != 'PyCFunction':
            func_ptr = '(void*)(%s)%s' % (cast, func_ptr)
        entry_name = entry.name.as_c_string_literal()
        if is_reverse_number_slot:
            slot = TypeSlots.get_slot_table(self.globalstate.directives).get_slot_by_method_name(entry.name)
            preproc_guard = slot.preprocessor_guard_code()
            if preproc_guard:
                self.putln(preproc_guard)
        self.putln('{%s, (PyCFunction)%s, %s, %s}%s' % (entry_name, func_ptr, '|'.join(method_flags), entry.doc_cname if entry.doc else '0', term))
        if is_reverse_number_slot and preproc_guard:
            self.putln('#endif')

    def put_pymethoddef_wrapper(self, entry):
        func_cname = entry.func_cname
        if entry.is_special:
            method_flags = entry.signature.method_flags() or []
            from .TypeSlots import method_noargs
            if method_noargs in method_flags:
                func_cname = Naming.method_wrapper_prefix + func_cname
                self.putln('static PyObject *%s(PyObject *self, CYTHON_UNUSED PyObject *arg) {' % func_cname)
                func_call = '%s(self)' % entry.func_cname
                if entry.name == '__next__':
                    self.putln('PyObject *res = %s;' % func_call)
                    self.putln('if (!res && !PyErr_Occurred()) { PyErr_SetNone(PyExc_StopIteration); }')
                    self.putln('return res;')
                else:
                    self.putln('return %s;' % func_call)
                self.putln('}')
        return func_cname

    def use_fast_gil_utility_code(self):
        if self.globalstate.directives['fast_gil']:
            self.globalstate.use_utility_code(UtilityCode.load_cached('FastGil', 'ModuleSetupCode.c'))
        else:
            self.globalstate.use_utility_code(UtilityCode.load_cached('NoFastGil', 'ModuleSetupCode.c'))

    def put_ensure_gil(self, declare_gilstate=True, variable=None):
        """
        Acquire the GIL. The generated code is safe even when no PyThreadState
        has been allocated for this thread (for threads not initialized by
        using the Python API). Additionally, the code generated by this method
        may be called recursively.
        """
        self.globalstate.use_utility_code(UtilityCode.load_cached('ForceInitThreads', 'ModuleSetupCode.c'))
        self.use_fast_gil_utility_code()
        self.putln('#ifdef WITH_THREAD')
        if not variable:
            variable = '__pyx_gilstate_save'
            if declare_gilstate:
                self.put('PyGILState_STATE ')
        self.putln('%s = __Pyx_PyGILState_Ensure();' % variable)
        self.putln('#endif')

    def put_release_ensured_gil(self, variable=None):
        """
        Releases the GIL, corresponds to `put_ensure_gil`.
        """
        self.use_fast_gil_utility_code()
        if not variable:
            variable = '__pyx_gilstate_save'
        self.putln('#ifdef WITH_THREAD')
        self.putln('__Pyx_PyGILState_Release(%s);' % variable)
        self.putln('#endif')

    def put_acquire_gil(self, variable=None, unknown_gil_state=True):
        """
        Acquire the GIL. The thread's thread state must have been initialized
        by a previous `put_release_gil`
        """
        self.use_fast_gil_utility_code()
        self.putln('#ifdef WITH_THREAD')
        self.putln('__Pyx_FastGIL_Forget();')
        if variable:
            self.putln('_save = %s;' % variable)
        if unknown_gil_state:
            self.putln('if (_save) {')
        self.putln('Py_BLOCK_THREADS')
        if unknown_gil_state:
            self.putln('}')
        self.putln('#endif')

    def put_release_gil(self, variable=None, unknown_gil_state=True):
        """Release the GIL, corresponds to `put_acquire_gil`."""
        self.use_fast_gil_utility_code()
        self.putln('#ifdef WITH_THREAD')
        self.putln('PyThreadState *_save;')
        self.putln('_save = NULL;')
        if unknown_gil_state:
            self.putln('if (PyGILState_Check()) {')
        self.putln('Py_UNBLOCK_THREADS')
        if unknown_gil_state:
            self.putln('}')
        if variable:
            self.putln('%s = _save;' % variable)
        self.putln('__Pyx_FastGIL_Remember();')
        self.putln('#endif')

    def declare_gilstate(self):
        self.putln('#ifdef WITH_THREAD')
        self.putln('PyGILState_STATE __pyx_gilstate_save;')
        self.putln('#endif')

    def put_error_if_neg(self, pos, value):
        return self.putln('if (%s < 0) %s' % (value, self.error_goto(pos)))

    def put_error_if_unbound(self, pos, entry, in_nogil_context=False, unbound_check_code=None):
        if entry.from_closure:
            func = '__Pyx_RaiseClosureNameError'
            self.globalstate.use_utility_code(UtilityCode.load_cached('RaiseClosureNameError', 'ObjectHandling.c'))
        elif entry.type.is_memoryviewslice and in_nogil_context:
            func = '__Pyx_RaiseUnboundMemoryviewSliceNogil'
            self.globalstate.use_utility_code(UtilityCode.load_cached('RaiseUnboundMemoryviewSliceNogil', 'ObjectHandling.c'))
        elif entry.type.is_cpp_class and entry.is_cglobal:
            func = '__Pyx_RaiseCppGlobalNameError'
            self.globalstate.use_utility_code(UtilityCode.load_cached('RaiseCppGlobalNameError', 'ObjectHandling.c'))
        elif entry.type.is_cpp_class and entry.is_variable and (not entry.is_member) and entry.scope.is_c_class_scope:
            func = '__Pyx_RaiseCppAttributeError'
            self.globalstate.use_utility_code(UtilityCode.load_cached('RaiseCppAttributeError', 'ObjectHandling.c'))
        else:
            func = '__Pyx_RaiseUnboundLocalError'
            self.globalstate.use_utility_code(UtilityCode.load_cached('RaiseUnboundLocalError', 'ObjectHandling.c'))
        if not unbound_check_code:
            unbound_check_code = entry.type.check_for_null_code(entry.cname)
        self.putln('if (unlikely(!%s)) { %s("%s"); %s }' % (unbound_check_code, func, entry.name, self.error_goto(pos)))

    def set_error_info(self, pos, used=False):
        self.funcstate.should_declare_error_indicator = True
        if used:
            self.funcstate.uses_error_indicator = True
        return '__PYX_MARK_ERR_POS(%s, %s)' % (self.lookup_filename(pos[0]), pos[1])

    def error_goto(self, pos, used=True):
        lbl = self.funcstate.error_label
        self.funcstate.use_label(lbl)
        if pos is None:
            return 'goto %s;' % lbl
        self.funcstate.should_declare_error_indicator = True
        if used:
            self.funcstate.uses_error_indicator = True
        return '__PYX_ERR(%s, %s, %s)' % (self.lookup_filename(pos[0]), pos[1], lbl)

    def error_goto_if(self, cond, pos):
        return 'if (%s) %s' % (self.unlikely(cond), self.error_goto(pos))

    def error_goto_if_null(self, cname, pos):
        return self.error_goto_if('!%s' % cname, pos)

    def error_goto_if_neg(self, cname, pos):
        return self.error_goto_if('(%s < 0)' % cname, pos)

    def error_goto_if_PyErr(self, pos):
        return self.error_goto_if('PyErr_Occurred()', pos)

    def lookup_filename(self, filename):
        return self.globalstate.lookup_filename(filename)

    def put_declare_refcount_context(self):
        self.putln('__Pyx_RefNannyDeclarations')

    def put_setup_refcount_context(self, name, acquire_gil=False):
        name = name.as_c_string_literal()
        if acquire_gil:
            self.globalstate.use_utility_code(UtilityCode.load_cached('ForceInitThreads', 'ModuleSetupCode.c'))
        self.putln('__Pyx_RefNannySetupContext(%s, %d);' % (name, acquire_gil and 1 or 0))

    def put_finish_refcount_context(self, nogil=False):
        self.putln('__Pyx_RefNannyFinishContextNogil()' if nogil else '__Pyx_RefNannyFinishContext();')

    def put_add_traceback(self, qualified_name, include_cline=True):
        """
        Build a Python traceback for propagating exceptions.

        qualified_name should be the qualified name of the function.
        """
        qualified_name = qualified_name.as_c_string_literal()
        format_tuple = (qualified_name, Naming.clineno_cname if include_cline else 0, Naming.lineno_cname, Naming.filename_cname)
        self.funcstate.uses_error_indicator = True
        self.putln('__Pyx_AddTraceback(%s, %s, %s, %s);' % format_tuple)

    def put_unraisable(self, qualified_name, nogil=False):
        """
        Generate code to print a Python warning for an unraisable exception.

        qualified_name should be the qualified name of the function.
        """
        format_tuple = (qualified_name, Naming.clineno_cname, Naming.lineno_cname, Naming.filename_cname, self.globalstate.directives['unraisable_tracebacks'], nogil)
        self.funcstate.uses_error_indicator = True
        self.putln('__Pyx_WriteUnraisable("%s", %s, %s, %s, %d, %d);' % format_tuple)
        self.globalstate.use_utility_code(UtilityCode.load_cached('WriteUnraisableException', 'Exceptions.c'))

    def put_trace_declarations(self):
        self.putln('__Pyx_TraceDeclarations')

    def put_trace_frame_init(self, codeobj=None):
        if codeobj:
            self.putln('__Pyx_TraceFrameInit(%s)' % codeobj)

    def put_trace_call(self, name, pos, nogil=False):
        self.putln('__Pyx_TraceCall("%s", %s[%s], %s, %d, %s);' % (name, Naming.filetable_cname, self.lookup_filename(pos[0]), pos[1], nogil, self.error_goto(pos)))

    def put_trace_exception(self):
        self.putln('__Pyx_TraceException();')

    def put_trace_return(self, retvalue_cname, nogil=False):
        self.putln('__Pyx_TraceReturn(%s, %d);' % (retvalue_cname, nogil))

    def putln_openmp(self, string):
        self.putln('#ifdef _OPENMP')
        self.putln(string)
        self.putln('#endif /* _OPENMP */')

    def undef_builtin_expect(self, cond):
        """
        Redefine the macros likely() and unlikely to no-ops, depending on
        condition 'cond'
        """
        self.putln('#if %s' % cond)
        self.putln('    #undef likely')
        self.putln('    #undef unlikely')
        self.putln('    #define likely(x)   (x)')
        self.putln('    #define unlikely(x) (x)')
        self.putln('#endif')

    def redef_builtin_expect(self, cond):
        self.putln('#if %s' % cond)
        self.putln('    #undef likely')
        self.putln('    #undef unlikely')
        self.putln('    #define likely(x)   __builtin_expect(!!(x), 1)')
        self.putln('    #define unlikely(x) __builtin_expect(!!(x), 0)')
        self.putln('#endif')