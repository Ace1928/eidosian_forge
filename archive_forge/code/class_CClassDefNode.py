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
class CClassDefNode(ClassDefNode):
    child_attrs = ['body']
    buffer_defaults_node = None
    buffer_defaults_pos = None
    typedef_flag = False
    api = False
    objstruct_name = None
    typeobj_name = None
    check_size = None
    decorators = None
    shadow = False

    @property
    def punycode_class_name(self):
        return punycodify_name(self.class_name)

    def buffer_defaults(self, env):
        if not hasattr(self, '_buffer_defaults'):
            from . import Buffer
            if self.buffer_defaults_node:
                self._buffer_defaults = Buffer.analyse_buffer_options(self.buffer_defaults_pos, env, [], self.buffer_defaults_node, need_complete=False)
            else:
                self._buffer_defaults = None
        return self._buffer_defaults

    def declare(self, env):
        if self.module_name and self.visibility != 'extern':
            module_path = self.module_name.split('.')
            home_scope = env.find_imported_module(module_path, self.pos)
            if not home_scope:
                return None
        else:
            home_scope = env
        self.entry = home_scope.declare_c_class(name=self.class_name, pos=self.pos, defining=0, implementing=0, module_name=self.module_name, base_type=None, objstruct_cname=self.objstruct_name, typeobj_cname=self.typeobj_name, visibility=self.visibility, typedef_flag=self.typedef_flag, check_size=self.check_size, api=self.api, buffer_defaults=self.buffer_defaults(env), shadow=self.shadow)
        if self.bases and len(self.bases.args) > 1:
            self.entry.type.multiple_bases = True

    def _handle_cclass_decorators(self, env):
        extra_directives = {}
        if not self.decorators:
            return extra_directives
        from . import ExprNodes
        remaining_decorators = []
        for original_decorator in self.decorators:
            decorator = original_decorator.decorator
            decorator_call = None
            if isinstance(decorator, ExprNodes.CallNode):
                decorator_call = decorator
                decorator = decorator.function
            known_name = Builtin.exprnode_to_known_standard_library_name(decorator, env)
            if known_name == 'functools.total_ordering':
                if decorator_call:
                    error(decorator_call.pos, 'total_ordering cannot be called.')
                extra_directives['total_ordering'] = True
                continue
            elif known_name == 'dataclasses.dataclass':
                args = None
                kwds = {}
                if decorator_call:
                    if isinstance(decorator_call, ExprNodes.SimpleCallNode):
                        args = decorator_call.args
                    else:
                        args = decorator_call.positional_args.args
                        kwds_ = decorator_call.keyword_args
                        if kwds_:
                            kwds = kwds_.as_python_dict()
                extra_directives[known_name] = (args, kwds)
                continue
            remaining_decorators.append(original_decorator)
        if remaining_decorators:
            error(remaining_decorators[0].pos, 'Cdef functions/classes cannot take arbitrary decorators.')
        self.decorators = remaining_decorators
        return extra_directives

    def analyse_declarations(self, env):
        if env.in_cinclude and (not self.objstruct_name):
            error(self.pos, "Object struct name specification required for C class defined in 'extern from' block")
        extra_directives = self._handle_cclass_decorators(env)
        self.base_type = None
        if self.module_name:
            self.module = None
            for module in env.cimported_modules:
                if module.name == self.module_name:
                    self.module = module
            if self.module is None:
                self.module = ModuleScope(self.module_name, None, env.context)
                self.module.has_extern_class = 1
                env.add_imported_module(self.module)
        if self.bases.args:
            base = self.bases.args[0]
            base_type = base.analyse_as_type(env)
            if base_type in (PyrexTypes.c_int_type, PyrexTypes.c_long_type, PyrexTypes.c_float_type):
                base_type = env.lookup(base_type.sign_and_name()).type
            if base_type is None:
                error(base.pos, "First base of '%s' is not an extension type" % self.class_name)
            elif base_type == PyrexTypes.py_object_type:
                base_class_scope = None
            elif not base_type.is_extension_type and (not (base_type.is_builtin_type and base_type.objstruct_cname)):
                error(base.pos, "'%s' is not an extension type" % base_type)
            elif not base_type.is_complete():
                error(base.pos, "Base class '%s' of type '%s' is incomplete" % (base_type.name, self.class_name))
            elif base_type.scope and base_type.scope.directives and base_type.is_final_type:
                error(base.pos, "Base class '%s' of type '%s' is final" % (base_type, self.class_name))
            elif base_type.is_builtin_type and base_type.name in ('tuple', 'bytes'):
                error(base.pos, "inheritance from PyVarObject types like '%s' is not currently supported" % base_type.name)
            else:
                self.base_type = base_type
            if env.directives.get('freelist', 0) > 0 and base_type != PyrexTypes.py_object_type:
                warning(self.pos, 'freelists cannot be used on subtypes, only the base class can manage them', 1)
        has_body = self.body is not None
        if has_body and self.base_type and (not self.base_type.scope):
            self.base_type.defered_declarations.append(lambda: self.analyse_declarations(env))
            return
        if self.module_name and self.visibility != 'extern':
            module_path = self.module_name.split('.')
            home_scope = env.find_imported_module(module_path, self.pos)
            if not home_scope:
                return
        else:
            home_scope = env
        if self.visibility == 'extern':
            if self.module_name == '__builtin__' and self.class_name in Builtin.builtin_types and (env.qualified_name[:8] != 'cpython.'):
                warning(self.pos, '%s already a builtin Cython type' % self.class_name, 1)
        self.entry = home_scope.declare_c_class(name=self.class_name, pos=self.pos, defining=has_body and self.in_pxd, implementing=has_body and (not self.in_pxd), module_name=self.module_name, base_type=self.base_type, objstruct_cname=self.objstruct_name, typeobj_cname=self.typeobj_name, check_size=self.check_size, visibility=self.visibility, typedef_flag=self.typedef_flag, api=self.api, buffer_defaults=self.buffer_defaults(env), shadow=self.shadow)
        if self.bases and len(self.bases.args) > 1:
            self.entry.type.multiple_bases = True
        if self.shadow:
            home_scope.lookup(self.class_name).as_variable = self.entry
        if home_scope is not env and self.visibility == 'extern':
            env.add_imported_entry(self.class_name, self.entry, self.pos)
        self.scope = scope = self.entry.type.scope
        if scope is not None:
            if extra_directives:
                scope.directives = env.directives.copy()
                scope.directives.update(extra_directives)
            else:
                scope.directives = env.directives
            if 'dataclasses.dataclass' in scope.directives:
                is_frozen = False
                dataclass_config = scope.directives['dataclasses.dataclass']
                if dataclass_config:
                    decorator_kwargs = dataclass_config[1]
                    frozen_flag = decorator_kwargs.get('frozen')
                    is_frozen = frozen_flag and frozen_flag.is_literal and frozen_flag.value
                scope.is_c_dataclass_scope = 'frozen' if is_frozen else True
        if self.doc and Options.docstrings:
            scope.doc = embed_position(self.pos, self.doc)
        if has_body:
            self.body.analyse_declarations(scope)
            dict_entry = self.scope.lookup_here('__dict__')
            if dict_entry and dict_entry.is_variable and (not scope.defined and (not scope.implemented)):
                dict_entry.getter_cname = self.scope.mangle_internal('__dict__getter')
                self.scope.declare_property('__dict__', dict_entry.doc, dict_entry.pos)
            if self.in_pxd:
                scope.defined = 1
            else:
                scope.implemented = 1
        if len(self.bases.args) > 1:
            if not has_body or self.in_pxd:
                error(self.bases.args[1].pos, 'Only declare first base in declaration.')
            for other_base in self.bases.args[1:]:
                if other_base.analyse_as_type(env):
                    error(other_base.pos, 'Only one extension type base class allowed.')
            self.entry.type.early_init = 0
            from . import ExprNodes
            self.type_init_args = ExprNodes.TupleNode(self.pos, args=[ExprNodes.IdentifierStringNode(self.pos, value=self.class_name), self.bases, ExprNodes.DictNode(self.pos, key_value_pairs=[])])
        elif self.base_type:
            self.entry.type.early_init = self.base_type.is_external or self.base_type.early_init
            self.type_init_args = None
        else:
            self.entry.type.early_init = 1
            self.type_init_args = None
        env.allocate_vtable_names(self.entry)
        for thunk in self.entry.type.defered_declarations:
            thunk()

    def analyse_expressions(self, env):
        if self.body:
            scope = self.entry.type.scope
            self.body = self.body.analyse_expressions(scope)
        if self.type_init_args:
            self.type_init_args.analyse_expressions(env)
        return self

    def generate_function_definitions(self, env, code):
        if self.body:
            self.generate_lambda_definitions(self.scope, code)
            self.body.generate_function_definitions(self.scope, code)

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        if not self.entry.type.early_init:
            bases = None
            if self.type_init_args:
                bases = code.funcstate.allocate_temp(PyrexTypes.py_object_type, manage_ref=True)
                self.type_init_args.generate_evaluation_code(code)
                code.putln('%s = PyTuple_GET_ITEM(%s, 1);' % (bases, self.type_init_args.result()))
                code.put_incref(bases, PyrexTypes.py_object_type)
                first_base = '((PyTypeObject*)PyTuple_GET_ITEM(%s, 0))' % bases
                trial_type = code.funcstate.allocate_temp(PyrexTypes.py_object_type, manage_ref=True)
                code.putln('%s = __Pyx_PyType_GetSlot(&PyType_Type, tp_new, newfunc)(&PyType_Type, %s, NULL);' % (trial_type, self.type_init_args.result()))
                code.putln(code.error_goto_if_null(trial_type, self.pos))
                code.put_gotref(trial_type, py_object_type)
                code.putln('if (__Pyx_PyType_GetSlot((PyTypeObject*) %s, tp_base, PyTypeObject*) != %s) {' % (trial_type, first_base))
                trial_type_base = '__Pyx_PyType_GetSlot((PyTypeObject*) %s, tp_base, PyTypeObject*)' % trial_type
                code.putln('__Pyx_TypeName base_name = __Pyx_PyType_GetName(%s);' % trial_type_base)
                code.putln('__Pyx_TypeName type_name = __Pyx_PyType_GetName(%s);' % first_base)
                code.putln('PyErr_Format(PyExc_TypeError, "best base \'" __Pyx_FMT_TYPENAME "\' must be equal to first base \'" __Pyx_FMT_TYPENAME "\'",')
                code.putln('             base_name, type_name);')
                code.putln('__Pyx_DECREF_TypeName(base_name);')
                code.putln('__Pyx_DECREF_TypeName(type_name);')
                code.putln(code.error_goto(self.pos))
                code.putln('}')
                code.put_decref_clear(trial_type, PyrexTypes.py_object_type)
                code.funcstate.release_temp(trial_type)
                self.type_init_args.generate_disposal_code(code)
                self.type_init_args.free_temps(code)
            self.generate_type_ready_code(self.entry, code, bases_tuple_cname=bases, check_heap_type_bases=True)
            if bases is not None:
                code.put_decref_clear(bases, PyrexTypes.py_object_type)
                code.funcstate.release_temp(bases)
        if self.body:
            self.body.generate_execution_code(code)

    @staticmethod
    def generate_type_ready_code(entry, code, bases_tuple_cname=None, check_heap_type_bases=False):
        type = entry.type
        typeptr_cname = type.typeptr_cname
        scope = type.scope
        if not scope:
            return
        if entry.visibility == 'extern':
            if type.typeobj_cname:
                assert not type.typeobj_cname
                code.putln('%s = &%s;' % (type.typeptr_cname, type.typeobj_cname))
            return
        else:
            assert typeptr_cname
            assert type.typeobj_cname
            typespec_cname = '%s_spec' % type.typeobj_cname
            code.putln('#if CYTHON_USE_TYPE_SPECS')
            tuple_temp = None
            if not bases_tuple_cname and scope.parent_type.base_type:
                tuple_temp = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
                code.putln('%s = PyTuple_Pack(1, (PyObject *)%s); %s' % (tuple_temp, scope.parent_type.base_type.typeptr_cname, code.error_goto_if_null(tuple_temp, entry.pos)))
                code.put_gotref(tuple_temp, py_object_type)
            if bases_tuple_cname or tuple_temp:
                if check_heap_type_bases:
                    code.globalstate.use_utility_code(UtilityCode.load_cached('ValidateBasesTuple', 'ExtensionTypes.c'))
                    code.put_error_if_neg(entry.pos, '__Pyx_validate_bases_tuple(%s.name, %s, %s)' % (typespec_cname, TypeSlots.get_slot_by_name('tp_dictoffset', scope.directives).slot_code(scope), bases_tuple_cname or tuple_temp))
                code.putln('%s = (PyTypeObject *) __Pyx_PyType_FromModuleAndSpec(%s, &%s, %s);' % (typeptr_cname, Naming.module_cname, typespec_cname, bases_tuple_cname or tuple_temp))
                if tuple_temp:
                    code.put_xdecref_clear(tuple_temp, type=py_object_type)
                    code.funcstate.release_temp(tuple_temp)
                code.putln(code.error_goto_if_null(typeptr_cname, entry.pos))
            else:
                code.putln('%s = (PyTypeObject *) __Pyx_PyType_FromModuleAndSpec(%s, &%s, NULL); %s' % (typeptr_cname, Naming.module_cname, typespec_cname, code.error_goto_if_null(typeptr_cname, entry.pos)))
            buffer_slot = TypeSlots.get_slot_by_name('tp_as_buffer', code.globalstate.directives)
            if not buffer_slot.is_empty(scope):
                code.putln('#if !CYTHON_COMPILING_IN_LIMITED_API')
                code.putln('%s->%s = %s;' % (typeptr_cname, buffer_slot.slot_name, buffer_slot.slot_code(scope)))
                for buffer_method_name in ('__getbuffer__', '__releasebuffer__'):
                    buffer_slot = TypeSlots.get_slot_table(code.globalstate.directives).get_slot_by_method_name(buffer_method_name)
                    if buffer_slot.slot_code(scope) == '0' and (not TypeSlots.get_base_slot_function(scope, buffer_slot)):
                        code.putln('if (!%s->tp_as_buffer->%s && %s->tp_base->tp_as_buffer && %s->tp_base->tp_as_buffer->%s) {' % (typeptr_cname, buffer_slot.slot_name, typeptr_cname, typeptr_cname, buffer_slot.slot_name))
                        code.putln('%s->tp_as_buffer->%s = %s->tp_base->tp_as_buffer->%s;' % (typeptr_cname, buffer_slot.slot_name, typeptr_cname, buffer_slot.slot_name))
                        code.putln('}')
                code.putln('#elif defined(Py_bf_getbuffer) && defined(Py_bf_releasebuffer)')
                code.putln('/* PY_VERSION_HEX >= 0x03090000 || Py_LIMITED_API >= 0x030B0000 */')
                code.putln('#elif defined(_MSC_VER)')
                code.putln('#pragma message ("The buffer protocol is not supported in the Limited C-API < 3.11.")')
                code.putln('#else')
                code.putln('#warning "The buffer protocol is not supported in the Limited C-API < 3.11."')
                code.putln('#endif')
            code.globalstate.use_utility_code(UtilityCode.load_cached('FixUpExtensionType', 'ExtensionTypes.c'))
            code.put_error_if_neg(entry.pos, '__Pyx_fix_up_extension_type_from_spec(&%s, %s)' % (typespec_cname, typeptr_cname))
            code.putln('#else')
            if bases_tuple_cname:
                code.put_incref(bases_tuple_cname, py_object_type)
                code.put_giveref(bases_tuple_cname, py_object_type)
                code.putln('%s.tp_bases = %s;' % (type.typeobj_cname, bases_tuple_cname))
            code.putln('%s = &%s;' % (typeptr_cname, type.typeobj_cname))
            code.putln('#endif')
            base_type = type.base_type
            while base_type:
                if base_type.is_external and (not base_type.objstruct_cname == 'PyTypeObject'):
                    code.putln('if (sizeof(%s%s) != sizeof(%s%s)) {' % ('' if type.typedef_flag else 'struct ', type.objstruct_cname, '' if base_type.typedef_flag else 'struct ', base_type.objstruct_cname))
                    code.globalstate.use_utility_code(UtilityCode.load_cached('ValidateExternBase', 'ExtensionTypes.c'))
                    code.put_error_if_neg(entry.pos, '__Pyx_validate_extern_base(%s)' % type.base_type.typeptr_cname)
                    code.putln('}')
                    break
                base_type = base_type.base_type
            code.putln('#if !CYTHON_COMPILING_IN_LIMITED_API')
            for slot in TypeSlots.get_slot_table(code.globalstate.directives):
                slot.generate_dynamic_init_code(scope, code)
            code.putln('#endif')
            code.putln('#if !CYTHON_USE_TYPE_SPECS')
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyType_Ready', 'ExtensionTypes.c'))
            code.put_error_if_neg(entry.pos, '__Pyx_PyType_Ready(%s)' % typeptr_cname)
            code.putln('#endif')
            code.putln('#if PY_MAJOR_VERSION < 3')
            code.putln('%s->tp_print = 0;' % typeptr_cname)
            code.putln('#endif')
            getattr_slot_func = TypeSlots.get_slot_code_by_name(scope, 'tp_getattro')
            dictoffset_slot_func = TypeSlots.get_slot_code_by_name(scope, 'tp_dictoffset')
            if getattr_slot_func == '0' and dictoffset_slot_func == '0':
                code.putln('#if !CYTHON_COMPILING_IN_LIMITED_API')
                if type.is_final_type:
                    py_cfunc = '__Pyx_PyObject_GenericGetAttrNoDict'
                    utility_func = 'PyObject_GenericGetAttrNoDict'
                else:
                    py_cfunc = '__Pyx_PyObject_GenericGetAttr'
                    utility_func = 'PyObject_GenericGetAttr'
                code.globalstate.use_utility_code(UtilityCode.load_cached(utility_func, 'ObjectHandling.c'))
                code.putln('if ((CYTHON_USE_TYPE_SLOTS && CYTHON_USE_PYTYPE_LOOKUP) && likely(!%s->tp_dictoffset && %s->tp_getattro == PyObject_GenericGetAttr)) {' % (typeptr_cname, typeptr_cname))
                code.putln('%s->tp_getattro = %s;' % (typeptr_cname, py_cfunc))
                code.putln('}')
                code.putln('#endif')
            for func in entry.type.scope.pyfunc_entries:
                is_buffer = func.name in ('__getbuffer__', '__releasebuffer__')
                if func.is_special and Options.docstrings and func.wrapperbase_cname and (not is_buffer):
                    slot = TypeSlots.get_slot_table(entry.type.scope.directives).get_slot_by_method_name(func.name)
                    preprocessor_guard = slot.preprocessor_guard_code() if slot else None
                    if preprocessor_guard:
                        code.putln(preprocessor_guard)
                    code.putln('#if CYTHON_UPDATE_DESCRIPTOR_DOC')
                    code.putln('{')
                    code.putln('PyObject *wrapper = PyObject_GetAttrString((PyObject *)%s, "%s"); %s' % (typeptr_cname, func.name, code.error_goto_if_null('wrapper', entry.pos)))
                    code.putln('if (__Pyx_IS_TYPE(wrapper, &PyWrapperDescr_Type)) {')
                    code.putln('%s = *((PyWrapperDescrObject *)wrapper)->d_base;' % func.wrapperbase_cname)
                    code.putln('%s.doc = %s;' % (func.wrapperbase_cname, func.doc_cname))
                    code.putln('((PyWrapperDescrObject *)wrapper)->d_base = &%s;' % func.wrapperbase_cname)
                    code.putln('}')
                    code.putln('}')
                    code.putln('#endif')
                    if preprocessor_guard:
                        code.putln('#endif')
            if type.vtable_cname:
                code.globalstate.use_utility_code(UtilityCode.load_cached('SetVTable', 'ImportExport.c'))
                code.put_error_if_neg(entry.pos, '__Pyx_SetVtable(%s, %s)' % (typeptr_cname, type.vtabptr_cname))
                code.putln('#if !CYTHON_COMPILING_IN_LIMITED_API')
                code.globalstate.use_utility_code(UtilityCode.load_cached('MergeVTables', 'ImportExport.c'))
                code.put_error_if_neg(entry.pos, '__Pyx_MergeVtables(%s)' % typeptr_cname)
                code.putln('#endif')
            if not type.scope.is_internal and (not type.scope.directives.get('internal')):
                code.put_error_if_neg(entry.pos, 'PyObject_SetAttr(%s, %s, (PyObject *) %s)' % (Naming.module_cname, code.intern_identifier(scope.class_name), typeptr_cname))
            weakref_entry = scope.lookup_here('__weakref__') if not scope.is_closure_class_scope else None
            if weakref_entry:
                if weakref_entry.type is py_object_type:
                    tp_weaklistoffset = '%s->tp_weaklistoffset' % typeptr_cname
                    if type.typedef_flag:
                        objstruct = type.objstruct_cname
                    else:
                        objstruct = 'struct %s' % type.objstruct_cname
                    code.putln('if (%s == 0) %s = offsetof(%s, %s);' % (tp_weaklistoffset, tp_weaklistoffset, objstruct, weakref_entry.cname))
                else:
                    error(weakref_entry.pos, "__weakref__ slot must be of type 'object'")
            if scope.lookup_here('__reduce_cython__') if not scope.is_closure_class_scope else None:
                code.globalstate.use_utility_code(UtilityCode.load_cached('SetupReduce', 'ExtensionTypes.c'))
                code.putln('#if !CYTHON_COMPILING_IN_LIMITED_API')
                code.put_error_if_neg(entry.pos, '__Pyx_setup_reduce((PyObject *) %s)' % typeptr_cname)
                code.putln('#endif')

    def annotate(self, code):
        if self.type_init_args:
            self.type_init_args.annotate(code)
        if self.body:
            self.body.annotate(code)