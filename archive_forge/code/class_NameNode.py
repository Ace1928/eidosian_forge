from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
class NameNode(AtomicExprNode):
    is_name = True
    is_cython_module = False
    cython_attribute = None
    lhs_of_first_assignment = False
    is_used_as_rvalue = 0
    entry = None
    type_entry = None
    cf_maybe_null = True
    cf_is_null = False
    allow_null = False
    nogil = False
    inferred_type = None

    def as_cython_attribute(self):
        return self.cython_attribute

    def type_dependencies(self, env):
        if self.entry is None:
            self.entry = env.lookup(self.name)
        if self.entry is not None and self.entry.type.is_unspecified:
            return (self,)
        else:
            return ()

    def infer_type(self, env):
        if self.entry is None:
            self.entry = env.lookup(self.name)
        if self.entry is None or self.entry.type is unspecified_type:
            if self.inferred_type is not None:
                return self.inferred_type
            return py_object_type
        elif (self.entry.type.is_extension_type or self.entry.type.is_builtin_type) and self.name == self.entry.type.name:
            return type_type
        elif self.entry.type.is_cfunction:
            if self.entry.scope.is_builtin_scope:
                return py_object_type
            else:
                return PyrexTypes.CPtrType(self.entry.type)
        else:
            if self.entry.type.is_pyobject and self.inferred_type:
                if not (self.inferred_type.is_int and self.entry.might_overflow):
                    return self.inferred_type
            return self.entry.type

    def compile_time_value(self, denv):
        try:
            return denv.lookup(self.name)
        except KeyError:
            error(self.pos, "Compile-time name '%s' not defined" % self.name)

    def get_constant_c_result_code(self):
        if not self.entry or self.entry.type.is_pyobject:
            return None
        return self.entry.cname

    def coerce_to(self, dst_type, env):
        if dst_type is py_object_type:
            entry = self.entry
            if entry and entry.is_cfunction:
                var_entry = entry.as_variable
                if var_entry:
                    if var_entry.is_builtin and var_entry.is_const:
                        var_entry = env.declare_builtin(var_entry.name, self.pos)
                    node = NameNode(self.pos, name=self.name)
                    node.entry = var_entry
                    node.analyse_rvalue_entry(env)
                    return node
        return super(NameNode, self).coerce_to(dst_type, env)

    def declare_from_annotation(self, env, as_target=False):
        """Implements PEP 526 annotation typing in a fairly relaxed way.

        Annotations are ignored for global variables.
        All other annotations are stored on the entry in the symbol table.
        String literals are allowed and not evaluated.
        The ambiguous Python types 'int' and 'long' are not evaluated - the 'cython.int' form must be used instead.
        """
        name = self.name
        annotation = self.annotation
        entry = self.entry or env.lookup_here(name)
        if not entry:
            if env.is_module_scope:
                return
            modifiers = ()
            if annotation.expr.is_string_literal or not env.directives['annotation_typing']:
                atype = None
            elif env.is_py_class_scope:
                atype = py_object_type
            else:
                modifiers, atype = annotation.analyse_type_annotation(env)
            if atype is None:
                atype = unspecified_type if as_target and env.directives['infer_types'] != False else py_object_type
            elif atype.is_fused and env.fused_to_specific:
                try:
                    atype = atype.specialize(env.fused_to_specific)
                except CannotSpecialize:
                    error(self.pos, "'%s' cannot be specialized since its type is not a fused argument to this function" % self.name)
                    atype = error_type
            visibility = 'private'
            if env.is_c_dataclass_scope:
                is_frozen = env.is_c_dataclass_scope == 'frozen'
                if atype.is_pyobject or atype.can_coerce_to_pyobject(env):
                    visibility = 'readonly' if is_frozen else 'public'
            if as_target and env.is_c_class_scope and (not (atype.is_pyobject or atype.is_error)):
                atype = py_object_type
                warning(annotation.pos, 'Annotation ignored since class-level attributes must be Python objects. Were you trying to set up an instance attribute?', 2)
            entry = self.entry = env.declare_var(name, atype, self.pos, is_cdef=not as_target, visibility=visibility, pytyping_modifiers=modifiers)
        if annotation and (not entry.annotation):
            entry.annotation = annotation

    def analyse_as_module(self, env):
        entry = self.entry
        if not entry:
            entry = env.lookup(self.name)
        if entry and entry.as_module:
            return entry.as_module
        if entry and entry.known_standard_library_import:
            scope = Builtin.get_known_standard_library_module_scope(entry.known_standard_library_import)
            if scope and scope.is_module_scope:
                return scope
        return None

    def analyse_as_type(self, env):
        type = None
        if self.cython_attribute:
            type = PyrexTypes.parse_basic_type(self.cython_attribute)
        elif env.in_c_type_context:
            type = PyrexTypes.parse_basic_type(self.name)
        if type:
            return type
        entry = self.entry
        if not entry:
            entry = env.lookup(self.name)
        if entry and (not entry.is_type) and entry.known_standard_library_import:
            entry = Builtin.get_known_standard_library_entry(entry.known_standard_library_import)
        if entry and entry.is_type:
            type = entry.type
            if not env.in_c_type_context and type is Builtin.long_type:
                warning(self.pos, "Found Python 2.x type 'long' in a Python annotation. Did you mean to use 'cython.long'?")
                type = py_object_type
            elif type.is_pyobject and type.equivalent_type:
                type = type.equivalent_type
            elif type is Builtin.int_type and env.global_scope().context.language_level == 2:
                type = py_object_type
            return type
        if self.name == 'object':
            return py_object_type
        if not env.in_c_type_context and PyrexTypes.parse_basic_type(self.name):
            warning(self.pos, "Found C type '%s' in a Python annotation. Did you mean to use 'cython.%s'?" % (self.name, self.name))
        return None

    def analyse_as_extension_type(self, env):
        entry = self.entry
        if not entry:
            entry = env.lookup(self.name)
        if entry and entry.is_type:
            if entry.type.is_extension_type or entry.type.is_builtin_type:
                return entry.type
        return None

    def analyse_target_declaration(self, env):
        return self._analyse_target_declaration(env, is_assignment_expression=False)

    def analyse_assignment_expression_target_declaration(self, env):
        return self._analyse_target_declaration(env, is_assignment_expression=True)

    def _analyse_target_declaration(self, env, is_assignment_expression):
        self.is_target = True
        if not self.entry:
            if is_assignment_expression:
                self.entry = env.lookup_assignment_expression_target(self.name)
            else:
                self.entry = env.lookup_here(self.name)
        if self.entry:
            self.entry.known_standard_library_import = ''
        if not self.entry and self.annotation is not None:
            is_dataclass = env.is_c_dataclass_scope
            self.declare_from_annotation(env, as_target=not is_dataclass)
        elif self.entry and self.entry.is_inherited and self.annotation and env.is_c_dataclass_scope:
            error(self.pos, 'Cannot redeclare inherited fields in Cython dataclasses')
        if not self.entry:
            if env.directives['warn.undeclared']:
                warning(self.pos, "implicit declaration of '%s'" % self.name, 1)
            if env.directives['infer_types'] != False:
                type = unspecified_type
            else:
                type = py_object_type
            if is_assignment_expression:
                self.entry = env.declare_assignment_expression_target(self.name, type, self.pos)
            else:
                self.entry = env.declare_var(self.name, type, self.pos)
        if self.entry.is_declared_generic:
            self.result_ctype = py_object_type
        if self.entry.as_module:
            self.entry.is_variable = 1

    def analyse_types(self, env):
        self.initialized_check = env.directives['initializedcheck']
        entry = self.entry
        if entry is None:
            entry = env.lookup(self.name)
            if not entry:
                entry = env.declare_builtin(self.name, self.pos)
                if entry and entry.is_builtin and entry.is_const:
                    self.is_literal = True
            if not entry:
                self.type = PyrexTypes.error_type
                return self
            self.entry = entry
        entry.used = 1
        if entry.type.is_buffer:
            from . import Buffer
            Buffer.used_buffer_aux_vars(entry)
        self.analyse_rvalue_entry(env)
        return self

    def analyse_target_types(self, env):
        self.analyse_entry(env, is_target=True)
        entry = self.entry
        if entry.is_cfunction and entry.as_variable:
            if (entry.is_overridable or entry.type.is_overridable) or (not self.is_lvalue() and entry.fused_cfunction):
                entry = self.entry = entry.as_variable
                self.type = entry.type
        if self.type.is_const:
            error(self.pos, "Assignment to const '%s'" % self.name)
        if not self.is_lvalue():
            error(self.pos, "Assignment to non-lvalue '%s'" % self.name)
            self.type = PyrexTypes.error_type
        entry.used = 1
        if entry.type.is_buffer:
            from . import Buffer
            Buffer.used_buffer_aux_vars(entry)
        return self

    def analyse_rvalue_entry(self, env):
        self.analyse_entry(env)
        entry = self.entry
        if entry.is_declared_generic:
            self.result_ctype = py_object_type
        if entry.is_pyglobal or entry.is_builtin:
            if entry.is_builtin and entry.is_const:
                self.is_temp = 0
            else:
                self.is_temp = 1
            self.is_used_as_rvalue = 1
        elif entry.type.is_memoryviewslice:
            self.is_temp = False
            self.is_used_as_rvalue = True
            self.use_managed_ref = True
        return self

    def nogil_check(self, env):
        self.nogil = True
        if self.is_used_as_rvalue:
            entry = self.entry
            if entry.is_builtin:
                if not entry.is_const:
                    self.gil_error()
            elif entry.is_pyglobal:
                self.gil_error()
    gil_message = 'Accessing Python global or builtin'

    def analyse_entry(self, env, is_target=False):
        self.check_identifier_kind()
        entry = self.entry
        type = entry.type
        if not is_target and type.is_pyobject and self.inferred_type and self.inferred_type.is_builtin_type:
            type = self.inferred_type
        self.type = type

    def check_identifier_kind(self):
        entry = self.entry
        if entry.is_type and entry.type.is_extension_type:
            self.type_entry = entry
        if entry.is_type and (entry.type.is_enum or entry.type.is_cpp_enum):
            py_entry = Symtab.Entry(self.name, None, py_object_type)
            py_entry.is_pyglobal = True
            py_entry.scope = self.entry.scope
            self.entry = py_entry
        elif not (entry.is_const or entry.is_variable or entry.is_builtin or entry.is_cfunction or entry.is_cpp_class):
            if self.entry.as_variable:
                self.entry = self.entry.as_variable
            elif not self.is_cython_module:
                error(self.pos, "'%s' is not a constant, variable or function identifier" % self.name)

    def is_cimported_module_without_shadow(self, env):
        if self.is_cython_module or self.cython_attribute:
            return False
        entry = self.entry or env.lookup(self.name)
        return entry.as_module and (not entry.is_variable)

    def is_simple(self):
        return 1

    def may_be_none(self):
        if self.cf_state and self.type and (self.type.is_pyobject or self.type.is_memoryviewslice):
            if getattr(self, '_none_checking', False):
                return False
            self._none_checking = True
            may_be_none = False
            for assignment in self.cf_state:
                if assignment.rhs.may_be_none():
                    may_be_none = True
                    break
            del self._none_checking
            return may_be_none
        return super(NameNode, self).may_be_none()

    def nonlocally_immutable(self):
        if ExprNode.nonlocally_immutable(self):
            return True
        entry = self.entry
        if not entry or entry.in_closure:
            return False
        return entry.is_local or entry.is_arg or entry.is_builtin or entry.is_readonly

    def calculate_target_results(self, env):
        pass

    def check_const(self):
        entry = self.entry
        if entry is not None and (not (entry.is_const or entry.is_cfunction or entry.is_builtin or entry.type.is_const)):
            self.not_const()
            return False
        return True

    def check_const_addr(self):
        entry = self.entry
        if not (entry.is_cglobal or entry.is_cfunction or entry.is_builtin):
            self.addr_not_const()
            return False
        return True

    def is_lvalue(self):
        return self.entry.is_variable and (not self.entry.is_readonly) or (self.entry.is_cfunction and self.entry.is_overridable)

    def is_addressable(self):
        return self.entry.is_variable and (not self.type.is_memoryviewslice)

    def is_ephemeral(self):
        return 0

    def calculate_result_code(self):
        entry = self.entry
        if not entry:
            return '<error>'
        if self.entry.is_cpp_optional and (not self.is_target):
            return '(*%s)' % entry.cname
        return entry.cname

    def generate_result_code(self, code):
        entry = self.entry
        if entry is None:
            return
        if entry.utility_code:
            code.globalstate.use_utility_code(entry.utility_code)
        if entry.is_builtin and entry.is_const:
            return
        elif entry.is_pyclass_attr:
            assert entry.type.is_pyobject, 'Python global or builtin not a Python object'
            interned_cname = code.intern_identifier(self.entry.name)
            if entry.is_builtin:
                namespace = Naming.builtins_cname
            else:
                namespace = entry.scope.namespace_cname
            if not self.cf_is_null:
                code.putln('%s = PyObject_GetItem(%s, %s);' % (self.result(), namespace, interned_cname))
                code.putln('if (unlikely(!%s)) {' % self.result())
                code.putln('PyErr_Clear();')
            code.globalstate.use_utility_code(UtilityCode.load_cached('GetModuleGlobalName', 'ObjectHandling.c'))
            code.putln('__Pyx_GetModuleGlobalName(%s, %s);' % (self.result(), interned_cname))
            if not self.cf_is_null:
                code.putln('}')
            code.putln(code.error_goto_if_null(self.result(), self.pos))
            self.generate_gotref(code)
        elif entry.is_builtin and (not entry.scope.is_module_scope):
            assert entry.type.is_pyobject, 'Python global or builtin not a Python object'
            interned_cname = code.intern_identifier(self.entry.name)
            code.globalstate.use_utility_code(UtilityCode.load_cached('GetBuiltinName', 'ObjectHandling.c'))
            code.putln('%s = __Pyx_GetBuiltinName(%s); %s' % (self.result(), interned_cname, code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
        elif entry.is_pyglobal or (entry.is_builtin and entry.scope.is_module_scope):
            assert entry.type.is_pyobject, 'Python global or builtin not a Python object'
            interned_cname = code.intern_identifier(self.entry.name)
            if entry.scope.is_module_scope:
                code.globalstate.use_utility_code(UtilityCode.load_cached('GetModuleGlobalName', 'ObjectHandling.c'))
                code.putln('__Pyx_GetModuleGlobalName(%s, %s); %s' % (self.result(), interned_cname, code.error_goto_if_null(self.result(), self.pos)))
            else:
                code.globalstate.use_utility_code(UtilityCode.load_cached('GetNameInClass', 'ObjectHandling.c'))
                code.putln('__Pyx_GetNameInClass(%s, %s, %s); %s' % (self.result(), entry.scope.namespace_cname, interned_cname, code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
        elif entry.is_local or entry.in_closure or entry.from_closure or entry.type.is_memoryviewslice:
            raise_unbound = (self.cf_maybe_null or self.cf_is_null) and (not self.allow_null)
            memslice_check = entry.type.is_memoryviewslice and self.initialized_check
            optional_cpp_check = entry.is_cpp_optional and self.initialized_check
            if optional_cpp_check:
                unbound_check_code = entry.type.cpp_optional_check_for_null_code(entry.cname)
            else:
                unbound_check_code = entry.type.check_for_null_code(entry.cname)
            if unbound_check_code and raise_unbound and (entry.type.is_pyobject or memslice_check or optional_cpp_check):
                code.put_error_if_unbound(self.pos, entry, self.in_nogil_context, unbound_check_code=unbound_check_code)
        elif entry.is_cglobal and entry.is_cpp_optional and self.initialized_check:
            unbound_check_code = entry.type.cpp_optional_check_for_null_code(entry.cname)
            code.put_error_if_unbound(self.pos, entry, unbound_check_code=unbound_check_code)

    def generate_assignment_code(self, rhs, code, overloaded_assignment=False, exception_check=None, exception_value=None):
        entry = self.entry
        if entry is None:
            return
        if self.entry.type.is_ptr and isinstance(rhs, ListNode) and (not self.lhs_of_first_assignment) and (not rhs.in_module_scope):
            error(self.pos, 'Literal list must be assigned to pointer at time of declaration')
        if entry.is_pyglobal:
            assert entry.type.is_pyobject, 'Python global or builtin not a Python object'
            interned_cname = code.intern_identifier(self.entry.name)
            namespace = self.entry.scope.namespace_cname
            if entry.is_member:
                setter = '__Pyx_SetItemOnTypeDict'
            elif entry.scope.is_module_scope:
                setter = 'PyDict_SetItem'
                namespace = Naming.moddict_cname
            elif entry.is_pyclass_attr:
                n = 'SetNewInClass' if self.name == '__new__' else 'SetNameInClass'
                code.globalstate.use_utility_code(UtilityCode.load_cached(n, 'ObjectHandling.c'))
                setter = '__Pyx_' + n
            else:
                assert False, repr(entry)
            code.put_error_if_neg(self.pos, '%s(%s, %s, %s)' % (setter, namespace, interned_cname, rhs.py_result()))
            if debug_disposal_code:
                print('NameNode.generate_assignment_code:')
                print('...generating disposal code for %s' % rhs)
            rhs.generate_disposal_code(code)
            rhs.free_temps(code)
            if entry.is_member:
                code.putln('PyType_Modified(%s);' % entry.scope.parent_type.typeptr_cname)
        else:
            if self.type.is_memoryviewslice:
                self.generate_acquire_memoryviewslice(rhs, code)
            elif self.type.is_buffer:
                self.generate_acquire_buffer(rhs, code)
            assigned = False
            if self.type.is_pyobject:
                if self.use_managed_ref:
                    rhs.make_owned_reference(code)
                    is_external_ref = entry.is_cglobal or self.entry.in_closure or self.entry.from_closure
                    if is_external_ref:
                        self.generate_gotref(code, handle_null=True)
                    assigned = True
                    if entry.is_cglobal:
                        self.generate_decref_set(code, rhs.result_as(self.ctype()))
                    elif not self.cf_is_null:
                        if self.cf_maybe_null:
                            self.generate_xdecref_set(code, rhs.result_as(self.ctype()))
                        else:
                            self.generate_decref_set(code, rhs.result_as(self.ctype()))
                    else:
                        assigned = False
                    if is_external_ref:
                        rhs.generate_giveref(code)
            if not self.type.is_memoryviewslice:
                if not assigned:
                    if overloaded_assignment:
                        result = rhs.move_result_rhs()
                        if exception_check == '+':
                            translate_cpp_exception(code, self.pos, '%s = %s;' % (self.result(), result), self.result() if self.type.is_pyobject else None, exception_value, self.in_nogil_context)
                        else:
                            code.putln('%s = %s;' % (self.result(), result))
                    else:
                        result = rhs.move_result_rhs_as(self.ctype())
                        if is_pythran_expr(self.type):
                            code.putln('new (&%s) decltype(%s){%s};' % (self.result(), self.result(), result))
                        elif result != self.result():
                            code.putln('%s = %s;' % (self.result(), result))
                if debug_disposal_code:
                    print('NameNode.generate_assignment_code:')
                    print('...generating post-assignment code for %s' % rhs)
                rhs.generate_post_assignment_code(code)
            elif rhs.result_in_temp():
                rhs.generate_post_assignment_code(code)
            rhs.free_temps(code)

    def generate_acquire_memoryviewslice(self, rhs, code):
        """
        Slices, coercions from objects, return values etc are new references.
        We have a borrowed reference in case of dst = src
        """
        from . import MemoryView
        MemoryView.put_acquire_memoryviewslice(lhs_cname=self.result(), lhs_type=self.type, lhs_pos=self.pos, rhs=rhs, code=code, have_gil=not self.in_nogil_context, first_assignment=self.cf_is_null)

    def generate_acquire_buffer(self, rhs, code):
        pretty_rhs = isinstance(rhs, NameNode) or rhs.is_temp
        if pretty_rhs:
            rhstmp = rhs.result_as(self.ctype())
        else:
            rhstmp = code.funcstate.allocate_temp(self.entry.type, manage_ref=False)
            code.putln('%s = %s;' % (rhstmp, rhs.result_as(self.ctype())))
        from . import Buffer
        Buffer.put_assign_to_buffer(self.result(), rhstmp, self.entry, is_initialized=not self.lhs_of_first_assignment, pos=self.pos, code=code)
        if not pretty_rhs:
            code.putln('%s = 0;' % rhstmp)
            code.funcstate.release_temp(rhstmp)

    def generate_deletion_code(self, code, ignore_nonexisting=False):
        if self.entry is None:
            return
        elif self.entry.is_pyclass_attr:
            namespace = self.entry.scope.namespace_cname
            interned_cname = code.intern_identifier(self.entry.name)
            if ignore_nonexisting:
                key_error_code = 'PyErr_Clear(); else'
            else:
                key_error_code = '{ PyErr_Clear(); PyErr_Format(PyExc_NameError, "name \'%%s\' is not defined", "%s"); }' % self.entry.name
            code.putln('if (unlikely(PyObject_DelItem(%s, %s) < 0)) { if (likely(PyErr_ExceptionMatches(PyExc_KeyError))) %s %s }' % (namespace, interned_cname, key_error_code, code.error_goto(self.pos)))
        elif self.entry.is_pyglobal:
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectSetAttrStr', 'ObjectHandling.c'))
            interned_cname = code.intern_identifier(self.entry.name)
            del_code = '__Pyx_PyObject_DelAttrStr(%s, %s)' % (Naming.module_cname, interned_cname)
            if ignore_nonexisting:
                code.putln('if (unlikely(%s < 0)) { if (likely(PyErr_ExceptionMatches(PyExc_AttributeError))) PyErr_Clear(); else %s }' % (del_code, code.error_goto(self.pos)))
            else:
                code.put_error_if_neg(self.pos, del_code)
        elif self.entry.type.is_pyobject or self.entry.type.is_memoryviewslice:
            if not self.cf_is_null:
                if self.cf_maybe_null and (not ignore_nonexisting):
                    code.put_error_if_unbound(self.pos, self.entry)
                if self.entry.in_closure:
                    self.generate_gotref(code, handle_null=True, maybe_null_extra_check=ignore_nonexisting)
                if ignore_nonexisting and self.cf_maybe_null:
                    code.put_xdecref_clear(self.result(), self.ctype(), have_gil=not self.nogil)
                else:
                    code.put_decref_clear(self.result(), self.ctype(), have_gil=not self.nogil)
        else:
            error(self.pos, 'Deletion of C names not supported')

    def annotate(self, code):
        if getattr(self, 'is_called', False):
            pos = (self.pos[0], self.pos[1], self.pos[2] - len(self.name) - 1)
            if self.type.is_pyobject:
                style, text = ('py_call', 'python function (%s)')
            else:
                style, text = ('c_call', 'c function (%s)')
            code.annotate(pos, AnnotationItem(style, text % self.type, size=len(self.name)))

    def get_known_standard_library_import(self):
        if self.entry:
            return self.entry.known_standard_library_import
        return None