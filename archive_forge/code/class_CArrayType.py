from __future__ import absolute_import
import copy
import hashlib
import re
from functools import partial
from itertools import product
from Cython.Utils import cached_function
from .Code import UtilityCode, LazyUtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from .Errors import error, CannotSpecialize, performance_hint
class CArrayType(CPointerBaseType):
    is_array = 1
    to_tuple_function = None

    def __init__(self, base_type, size):
        super(CArrayType, self).__init__(base_type)
        self.size = size

    def __eq__(self, other):
        if isinstance(other, CType) and other.is_array and (self.size == other.size):
            return self.base_type.same_as(other.base_type)
        return False

    def __hash__(self):
        return hash(self.base_type) + 28

    def __repr__(self):
        return '<CArrayType %s %s>' % (self.size, repr(self.base_type))

    def same_as_resolved_type(self, other_type):
        return other_type.is_array and self.base_type.same_as(other_type.base_type) or other_type is error_type

    def assignable_from_resolved_type(self, src_type):
        if src_type.is_pyobject:
            return True
        if src_type.is_ptr or src_type.is_array:
            return self.base_type.assignable_from(src_type.base_type)
        return False

    def element_ptr_type(self):
        return c_ptr_type(self.base_type)

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        if self.size is not None:
            dimension_code = self.size
        else:
            dimension_code = ''
        if entity_code.startswith('*'):
            entity_code = '(%s)' % entity_code
        return self.base_type.declaration_code('%s[%s]' % (entity_code, dimension_code), for_display, dll_linkage, pyrex)

    def as_argument_type(self):
        return c_ptr_type(self.base_type)

    def is_complete(self):
        return self.size is not None

    def specialize(self, values):
        base_type = self.base_type.specialize(values)
        if base_type == self.base_type:
            return self
        else:
            return CArrayType(base_type, self.size)

    def deduce_template_params(self, actual):
        if isinstance(actual, CArrayType):
            return self.base_type.deduce_template_params(actual.base_type)
        else:
            return {}

    def can_coerce_to_pyobject(self, env):
        return self.base_type.can_coerce_to_pyobject(env)

    def can_coerce_from_pyobject(self, env):
        return self.base_type.can_coerce_from_pyobject(env)

    def create_to_py_utility_code(self, env):
        if self.to_py_function is not None:
            return self.to_py_function
        if not self.base_type.create_to_py_utility_code(env):
            return False
        safe_typename = self.base_type.specialization_name()
        to_py_function = '__Pyx_carray_to_py_%s' % safe_typename
        to_tuple_function = '__Pyx_carray_to_tuple_%s' % safe_typename
        from .UtilityCode import CythonUtilityCode
        context = {'cname': to_py_function, 'to_tuple_cname': to_tuple_function, 'base_type': self.base_type}
        env.use_utility_code(CythonUtilityCode.load('carray.to_py', 'CConvert.pyx', outer_module_scope=env.global_scope(), context=context, compiler_directives=dict(env.global_scope().directives)))
        self.to_tuple_function = to_tuple_function
        self.to_py_function = to_py_function
        return True

    def to_py_call_code(self, source_code, result_code, result_type, to_py_function=None):
        func = self.to_py_function if to_py_function is None else to_py_function
        if self.is_string or self.is_pyunicode_ptr:
            return '%s = %s(%s)' % (result_code, func, source_code)
        target_is_tuple = result_type.is_builtin_type and result_type.name == 'tuple'
        return '%s = %s(%s, %s)' % (result_code, self.to_tuple_function if target_is_tuple else func, source_code, self.size)

    def create_from_py_utility_code(self, env):
        if self.from_py_function is not None:
            return self.from_py_function
        if not self.base_type.create_from_py_utility_code(env):
            return False
        from_py_function = '__Pyx_carray_from_py_%s' % self.base_type.specialization_name()
        from .UtilityCode import CythonUtilityCode
        context = {'cname': from_py_function, 'base_type': self.base_type}
        env.use_utility_code(CythonUtilityCode.load('carray.from_py', 'CConvert.pyx', outer_module_scope=env.global_scope(), context=context, compiler_directives=dict(env.global_scope().directives)))
        self.from_py_function = from_py_function
        return True

    def from_py_call_code(self, source_code, result_code, error_pos, code, from_py_function=None, error_condition=None, special_none_cvalue=None):
        assert not error_condition, '%s: %s' % (error_pos, error_condition)
        assert not special_none_cvalue, '%s: %s' % (error_pos, special_none_cvalue)
        call_code = '%s(%s, %s, %s)' % (from_py_function or self.from_py_function, source_code, result_code, self.size)
        return code.error_goto_if_neg(call_code, error_pos)

    def error_condition(self, result_code):
        return ''