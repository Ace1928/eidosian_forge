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
class AttributeNode(ExprNode):
    is_attribute = 1
    subexprs = ['obj']
    entry = None
    is_called = 0
    needs_none_check = True
    is_memslice_transpose = False
    is_special_lookup = False
    is_py_attr = 0

    def as_cython_attribute(self):
        if isinstance(self.obj, NameNode) and self.obj.is_cython_module and (not self.attribute == u'parallel'):
            return self.attribute
        cy = self.obj.as_cython_attribute()
        if cy:
            return '%s.%s' % (cy, self.attribute)
        return None

    def coerce_to(self, dst_type, env):
        if dst_type is py_object_type:
            entry = self.entry
            if entry and entry.is_cfunction and entry.as_variable:
                self.is_temp = 1
                self.entry = entry.as_variable
                self.analyse_as_python_attribute(env)
                return self
            elif entry and entry.is_cfunction and (self.obj.type is not Builtin.type_type):
                from .UtilNodes import EvalWithTempExprNode, ResultRefNode
                obj_node = ResultRefNode(self.obj, type=self.obj.type)
                obj_node.result_ctype = self.obj.result_ctype
                self.obj = obj_node
                unbound_node = ExprNode.coerce_to(self, dst_type, env)
                utility_code = UtilityCode.load_cached('PyMethodNew2Arg', 'ObjectHandling.c')
                func_type = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('func', PyrexTypes.py_object_type, None), PyrexTypes.CFuncTypeArg('self', PyrexTypes.py_object_type, None)])
                binding_call = PythonCapiCallNode(self.pos, function_name='__Pyx_PyMethod_New2Arg', func_type=func_type, args=[unbound_node, obj_node], utility_code=utility_code)
                complete_call = EvalWithTempExprNode(obj_node, binding_call)
                return complete_call.analyse_types(env)
        return ExprNode.coerce_to(self, dst_type, env)

    def calculate_constant_result(self):
        attr = self.attribute
        if attr.startswith('__') and attr.endswith('__'):
            return
        self.constant_result = getattr(self.obj.constant_result, attr)

    def compile_time_value(self, denv):
        attr = self.attribute
        if attr.startswith('__') and attr.endswith('__'):
            error(self.pos, "Invalid attribute name '%s' in compile-time expression" % attr)
            return None
        obj = self.obj.compile_time_value(denv)
        try:
            return getattr(obj, attr)
        except Exception as e:
            self.compile_time_value_error(e)

    def type_dependencies(self, env):
        return self.obj.type_dependencies(env)

    def infer_type(self, env):
        node = self.analyse_as_cimported_attribute_node(env, target=False)
        if node is not None:
            if node.entry.type and node.entry.type.is_cfunction:
                return PyrexTypes.CPtrType(node.entry.type)
            else:
                return node.entry.type
        node = self.analyse_as_type_attribute(env)
        if node is not None:
            return node.entry.type
        obj_type = self.obj.infer_type(env)
        self.analyse_attribute(env, obj_type=obj_type)
        if obj_type.is_builtin_type and self.type.is_cfunction:
            return py_object_type
        elif self.entry and self.entry.is_cmethod:
            return py_object_type
        return self.type

    def analyse_target_declaration(self, env):
        self.is_target = True

    def analyse_target_types(self, env):
        node = self.analyse_types(env, target=1)
        if node.type.is_const:
            error(self.pos, "Assignment to const attribute '%s'" % self.attribute)
        if not node.is_lvalue():
            error(self.pos, "Assignment to non-lvalue of type '%s'" % self.type)
        return node

    def analyse_types(self, env, target=0):
        if not self.type:
            self.type = PyrexTypes.error_type
        self.initialized_check = env.directives['initializedcheck']
        node = self.analyse_as_cimported_attribute_node(env, target)
        if node is None and (not target):
            node = self.analyse_as_type_attribute(env)
        if node is None:
            node = self.analyse_as_ordinary_attribute_node(env, target)
            assert node is not None
        if (node.is_attribute or node.is_name) and node.entry:
            node.entry.used = True
        if node.is_attribute:
            node.wrap_obj_in_nonecheck(env)
        return node

    def analyse_as_cimported_attribute_node(self, env, target):
        module_scope = self.obj.analyse_as_module(env)
        if module_scope:
            entry = module_scope.lookup_here(self.attribute)
            if entry and (not entry.known_standard_library_import) and (entry.is_cglobal or entry.is_cfunction or entry.is_type or entry.is_const):
                return self.as_name_node(env, entry, target)
            if self.is_cimported_module_without_shadow(env):
                error(self.pos, "cimported module has no attribute '%s'" % self.attribute)
                return self
        return None

    def analyse_as_type_attribute(self, env):
        if self.obj.is_string_literal:
            return
        type = self.obj.analyse_as_type(env)
        if type:
            if type.is_extension_type or type.is_builtin_type or type.is_cpp_class:
                entry = type.scope.lookup_here(self.attribute)
                if entry and (entry.is_cmethod or (type.is_cpp_class and entry.type.is_cfunction)):
                    if type.is_builtin_type:
                        if not self.is_called:
                            return None
                        ubcm_entry = entry
                    else:
                        ubcm_entry = self._create_unbound_cmethod_entry(type, entry, env)
                        ubcm_entry.overloaded_alternatives = [self._create_unbound_cmethod_entry(type, overloaded_alternative, env) for overloaded_alternative in entry.overloaded_alternatives]
                    return self.as_name_node(env, ubcm_entry, target=False)
            elif type.is_enum or type.is_cpp_enum:
                if self.attribute in type.values:
                    for entry in type.entry.enum_values:
                        if entry.name == self.attribute:
                            return self.as_name_node(env, entry, target=False)
                    else:
                        error(self.pos, '%s not a known value of %s' % (self.attribute, type))
                else:
                    error(self.pos, '%s not a known value of %s' % (self.attribute, type))
        return None

    def _create_unbound_cmethod_entry(self, type, entry, env):
        if entry.func_cname and entry.type.op_arg_struct is None:
            cname = entry.func_cname
            if entry.type.is_static_method or (env.parent_scope and env.parent_scope.is_cpp_class_scope):
                ctype = entry.type
            elif type.is_cpp_class:
                error(self.pos, '%s not a static member of %s' % (entry.name, type))
                ctype = PyrexTypes.error_type
            else:
                ctype = copy.copy(entry.type)
                ctype.args = ctype.args[:]
                ctype.args[0] = PyrexTypes.CFuncTypeArg('self', type, 'self', None)
        else:
            cname = '%s->%s' % (type.vtabptr_cname, entry.cname)
            ctype = entry.type
        ubcm_entry = Symtab.Entry(entry.name, cname, ctype)
        ubcm_entry.is_cfunction = 1
        ubcm_entry.func_cname = entry.func_cname
        ubcm_entry.is_unbound_cmethod = 1
        ubcm_entry.scope = entry.scope
        return ubcm_entry

    def analyse_as_type(self, env):
        module_scope = self.obj.analyse_as_module(env)
        if module_scope:
            return module_scope.lookup_type(self.attribute)
        if not self.obj.is_string_literal:
            base_type = self.obj.analyse_as_type(env)
            if base_type and getattr(base_type, 'scope', None) is not None:
                return base_type.scope.lookup_type(self.attribute)
        return None

    def analyse_as_extension_type(self, env):
        module_scope = self.obj.analyse_as_module(env)
        if module_scope:
            entry = module_scope.lookup_here(self.attribute)
            if entry and entry.is_type:
                if entry.type.is_extension_type or entry.type.is_builtin_type:
                    return entry.type
        return None

    def analyse_as_module(self, env):
        module_scope = self.obj.analyse_as_module(env)
        if module_scope:
            entry = module_scope.lookup_here(self.attribute)
            if entry and entry.as_module:
                return entry.as_module
        return None

    def as_name_node(self, env, entry, target):
        node = NameNode.from_node(self, name=self.attribute, entry=entry)
        if target:
            node = node.analyse_target_types(env)
        else:
            node = node.analyse_rvalue_entry(env)
        node.entry.used = 1
        return node

    def analyse_as_ordinary_attribute_node(self, env, target):
        self.obj = self.obj.analyse_types(env)
        self.analyse_attribute(env)
        if self.entry and self.entry.is_cmethod and (not self.is_called):
            pass
        if self.is_py_attr:
            if not target:
                self.is_temp = 1
                self.result_ctype = py_object_type
        elif target and self.obj.type.is_builtin_type:
            error(self.pos, 'Assignment to an immutable object field')
        elif self.entry and self.entry.is_cproperty:
            if not target:
                return SimpleCallNode.for_cproperty(self.pos, self.obj, self.entry).analyse_types(env)
            error(self.pos, 'Assignment to a read-only property')
        return self

    def analyse_attribute(self, env, obj_type=None):
        immutable_obj = obj_type is not None
        self.is_py_attr = 0
        self.member = self.attribute
        if obj_type is None:
            if self.obj.type.is_string or self.obj.type.is_pyunicode_ptr:
                self.obj = self.obj.coerce_to_pyobject(env)
            obj_type = self.obj.type
        elif obj_type.is_string or obj_type.is_pyunicode_ptr:
            obj_type = py_object_type
        if obj_type.is_ptr or obj_type.is_array:
            obj_type = obj_type.base_type
            self.op = '->'
        elif obj_type.is_extension_type or obj_type.is_builtin_type:
            self.op = '->'
        elif obj_type.is_reference and obj_type.is_fake_reference:
            self.op = '->'
        else:
            self.op = '.'
        if obj_type.has_attributes:
            if obj_type.attributes_known():
                entry = obj_type.scope.lookup_here(self.attribute)
                if obj_type.is_memoryviewslice and (not entry):
                    if self.attribute == 'T':
                        self.is_memslice_transpose = True
                        self.is_temp = True
                        self.use_managed_ref = True
                        self.type = self.obj.type.transpose(self.pos)
                        return
                    else:
                        obj_type.declare_attribute(self.attribute, env, self.pos)
                        entry = obj_type.scope.lookup_here(self.attribute)
                if entry and entry.is_member:
                    entry = None
            else:
                error(self.pos, "Cannot select attribute of incomplete type '%s'" % obj_type)
                self.type = PyrexTypes.error_type
                return
            self.entry = entry
            if entry:
                if obj_type.is_extension_type and entry.name == '__weakref__':
                    error(self.pos, 'Illegal use of special attribute __weakref__')
                if entry.is_cproperty:
                    self.type = entry.type
                    return
                elif entry.is_variable and (not entry.fused_cfunction) or entry.is_cmethod:
                    self.type = entry.type
                    self.member = entry.cname
                    return
                else:
                    pass
        self.analyse_as_python_attribute(env, obj_type, immutable_obj)

    def analyse_as_python_attribute(self, env, obj_type=None, immutable_obj=False):
        if obj_type is None:
            obj_type = self.obj.type
        self.attribute = env.mangle_class_private_name(self.attribute)
        self.member = self.attribute
        self.type = py_object_type
        self.is_py_attr = 1
        if not obj_type.is_pyobject and (not obj_type.is_error):
            if obj_type.is_string or obj_type.is_cpp_string or obj_type.is_buffer or obj_type.is_memoryviewslice or obj_type.is_numeric or (obj_type.is_ctuple and obj_type.can_coerce_to_pyobject(env)) or (obj_type.is_struct and obj_type.can_coerce_to_pyobject(env)):
                if not immutable_obj:
                    self.obj = self.obj.coerce_to_pyobject(env)
            elif obj_type.is_cfunction and (self.obj.is_name or self.obj.is_attribute) and self.obj.entry.as_variable and self.obj.entry.as_variable.type.is_pyobject:
                if not immutable_obj:
                    self.obj = self.obj.coerce_to_pyobject(env)
            else:
                error(self.pos, "Object of type '%s' has no attribute '%s'" % (obj_type, self.attribute))

    def wrap_obj_in_nonecheck(self, env):
        if not env.directives['nonecheck']:
            return
        msg = None
        format_args = ()
        if self.obj.type.is_extension_type and self.needs_none_check and (not self.is_py_attr):
            msg = "'NoneType' object has no attribute '%{0}s'".format('.30' if len(self.attribute) <= 30 else '')
            format_args = (self.attribute,)
        elif self.obj.type.is_memoryviewslice:
            if self.is_memslice_transpose:
                msg = 'Cannot transpose None memoryview slice'
            else:
                entry = self.obj.type.scope.lookup_here(self.attribute)
                if entry:
                    msg = "Cannot access '%s' attribute of None memoryview slice"
                    format_args = (entry.name,)
        if msg:
            self.obj = self.obj.as_none_safe_node(msg, 'PyExc_AttributeError', format_args=format_args)

    def nogil_check(self, env):
        if self.is_py_attr:
            self.gil_error()
    gil_message = 'Accessing Python attribute'

    def is_cimported_module_without_shadow(self, env):
        return self.obj.is_cimported_module_without_shadow(env)

    def is_simple(self):
        if self.obj:
            return self.result_in_temp() or self.obj.is_simple()
        else:
            return NameNode.is_simple(self)

    def is_lvalue(self):
        if self.obj:
            return True
        else:
            return NameNode.is_lvalue(self)

    def is_ephemeral(self):
        if self.obj:
            return self.obj.is_ephemeral()
        else:
            return NameNode.is_ephemeral(self)

    def calculate_result_code(self):
        result = self.calculate_access_code()
        if self.entry and self.entry.is_cpp_optional and (not self.is_target):
            result = '(*%s)' % result
        return result

    def calculate_access_code(self):
        obj = self.obj
        obj_code = obj.result_as(obj.type)
        if self.entry and self.entry.is_cmethod:
            if obj.type.is_extension_type and (not self.entry.is_builtin_cmethod):
                if self.entry.final_func_cname:
                    return self.entry.final_func_cname
                if self.type.from_fused:
                    self.member = self.entry.cname
                return '((struct %s *)%s%s%s)->%s' % (obj.type.vtabstruct_cname, obj_code, self.op, obj.type.vtabslot_cname, self.member)
            elif self.result_is_used:
                return self.member
            return
        elif obj.type.is_complex:
            return '__Pyx_C%s(%s)' % (self.member.upper(), obj_code)
        else:
            if obj.type.is_builtin_type and self.entry and self.entry.is_variable:
                obj_code = obj.type.cast_code(obj.result(), to_object_struct=True)
            return '%s%s%s' % (obj_code, self.op, self.member)

    def generate_result_code(self, code):
        if self.is_py_attr:
            if self.is_special_lookup:
                code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectLookupSpecial', 'ObjectHandling.c'))
                lookup_func_name = '__Pyx_PyObject_LookupSpecial'
            else:
                code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectGetAttrStr', 'ObjectHandling.c'))
                lookup_func_name = '__Pyx_PyObject_GetAttrStr'
            code.putln('%s = %s(%s, %s); %s' % (self.result(), lookup_func_name, self.obj.py_result(), code.intern_identifier(self.attribute), code.error_goto_if_null(self.result(), self.pos)))
            self.generate_gotref(code)
        elif self.type.is_memoryviewslice:
            if self.is_memslice_transpose:
                for access, packing in self.type.axes:
                    if access == 'ptr':
                        error(self.pos, 'Transposing not supported for slices with indirect dimensions')
                        return
                code.putln('%s = %s;' % (self.result(), self.obj.result()))
                code.put_incref_memoryviewslice(self.result(), self.type, have_gil=True)
                T = '__pyx_memslice_transpose(&%s)' % self.result()
                code.putln(code.error_goto_if_neg(T, self.pos))
            elif self.initialized_check:
                code.putln('if (unlikely(!%s.memview)) {PyErr_SetString(PyExc_AttributeError,"Memoryview is not initialized");%s}' % (self.result(), code.error_goto(self.pos)))
        elif self.entry.is_cpp_optional and self.initialized_check:
            if self.is_target:
                undereferenced_result = self.result()
            else:
                assert not self.is_temp
                undereferenced_result = self.calculate_access_code()
            unbound_check_code = self.type.cpp_optional_check_for_null_code(undereferenced_result)
            code.put_error_if_unbound(self.pos, self.entry, unbound_check_code=unbound_check_code)
        elif self.obj.type and self.obj.type.is_extension_type:
            pass
        elif self.entry and self.entry.is_cmethod:
            code.globalstate.use_entry_utility_code(self.entry)

    def generate_disposal_code(self, code):
        if self.is_temp and self.type.is_memoryviewslice and self.is_memslice_transpose:
            code.put_xdecref_clear(self.result(), self.type, have_gil=True)
        else:
            ExprNode.generate_disposal_code(self, code)

    def generate_assignment_code(self, rhs, code, overloaded_assignment=False, exception_check=None, exception_value=None):
        self.obj.generate_evaluation_code(code)
        if self.is_py_attr:
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectSetAttrStr', 'ObjectHandling.c'))
            code.put_error_if_neg(self.pos, '__Pyx_PyObject_SetAttrStr(%s, %s, %s)' % (self.obj.py_result(), code.intern_identifier(self.attribute), rhs.py_result()))
            rhs.generate_disposal_code(code)
            rhs.free_temps(code)
        elif self.obj.type.is_complex:
            code.putln('__Pyx_SET_C%s%s(%s, %s);' % (self.member.upper(), self.obj.type.implementation_suffix, self.obj.result_as(self.obj.type), rhs.result_as(self.ctype())))
            rhs.generate_disposal_code(code)
            rhs.free_temps(code)
        else:
            select_code = self.result()
            if self.type.is_pyobject and self.use_managed_ref:
                rhs.make_owned_reference(code)
                rhs.generate_giveref(code)
                code.put_gotref(select_code, self.type)
                code.put_decref(select_code, self.ctype())
            elif self.type.is_memoryviewslice:
                from . import MemoryView
                MemoryView.put_assign_to_memviewslice(select_code, rhs, rhs.result(), self.type, code)
            if not self.type.is_memoryviewslice:
                code.putln('%s = %s;' % (select_code, rhs.move_result_rhs_as(self.ctype())))
            rhs.generate_post_assignment_code(code)
            rhs.free_temps(code)
        self.obj.generate_disposal_code(code)
        self.obj.free_temps(code)

    def generate_deletion_code(self, code, ignore_nonexisting=False):
        self.obj.generate_evaluation_code(code)
        if self.is_py_attr or (self.entry.scope.is_property_scope and u'__del__' in self.entry.scope.entries):
            code.globalstate.use_utility_code(UtilityCode.load_cached('PyObjectSetAttrStr', 'ObjectHandling.c'))
            code.put_error_if_neg(self.pos, '__Pyx_PyObject_DelAttrStr(%s, %s)' % (self.obj.py_result(), code.intern_identifier(self.attribute)))
        else:
            error(self.pos, 'Cannot delete C attribute of extension type')
        self.obj.generate_disposal_code(code)
        self.obj.free_temps(code)

    def annotate(self, code):
        if self.is_py_attr:
            style, text = ('py_attr', 'python attribute (%s)')
        else:
            style, text = ('c_attr', 'c attribute (%s)')
        code.annotate(self.pos, AnnotationItem(style, text % self.type, size=len(self.attribute)))

    def get_known_standard_library_import(self):
        module_name = self.obj.get_known_standard_library_import()
        if module_name:
            return StringEncoding.EncodedString('%s.%s' % (module_name, self.attribute))
        return None