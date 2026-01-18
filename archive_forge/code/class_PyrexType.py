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
class PyrexType(BaseType):
    is_pyobject = 0
    is_unspecified = 0
    is_extension_type = 0
    is_final_type = 0
    is_builtin_type = 0
    is_cython_builtin_type = 0
    is_numeric = 0
    is_int = 0
    is_float = 0
    is_complex = 0
    is_void = 0
    is_array = 0
    is_ptr = 0
    is_null_ptr = 0
    is_reference = 0
    is_fake_reference = 0
    is_rvalue_reference = 0
    is_const = 0
    is_volatile = 0
    is_cv_qualified = 0
    is_cfunction = 0
    is_struct_or_union = 0
    is_cpp_class = 0
    is_optional_cpp_class = 0
    python_type_constructor_name = None
    is_cpp_string = 0
    is_struct = 0
    is_enum = 0
    is_cpp_enum = False
    is_typedef = 0
    is_string = 0
    is_pyunicode_ptr = 0
    is_unicode_char = 0
    is_returncode = 0
    is_error = 0
    is_buffer = 0
    is_ctuple = 0
    is_memoryviewslice = 0
    is_pythran_expr = 0
    is_numpy_buffer = 0
    has_attributes = 0
    needs_cpp_construction = 0
    needs_refcounting = 0
    refcounting_needs_gil = True
    equivalent_type = None
    default_value = ''
    declaration_value = ''

    def resolve(self):
        return self

    def specialize(self, values):
        return self

    def literal_code(self, value):
        return str(value)

    def __str__(self):
        return self.declaration_code('', for_display=1).strip()

    def same_as(self, other_type, **kwds):
        return self.same_as_resolved_type(other_type.resolve(), **kwds)

    def same_as_resolved_type(self, other_type):
        return self == other_type or other_type is error_type

    def subtype_of(self, other_type):
        return self.subtype_of_resolved_type(other_type.resolve())

    def subtype_of_resolved_type(self, other_type):
        return self.same_as(other_type)

    def assignable_from(self, src_type):
        return self.assignable_from_resolved_type(src_type.resolve())

    def assignable_from_resolved_type(self, src_type):
        return self.same_as(src_type)

    def assignment_failure_extra_info(self, src_type, src_name):
        """Override if you can provide useful extra information about why an assignment didn't work.

        src_name may be None if unavailable"""
        return ''

    def as_argument_type(self):
        return self

    def is_complete(self):
        return 1

    def is_simple_buffer_dtype(self):
        return False

    def can_be_optional(self):
        """Returns True if type can be used with typing.Optional[]."""
        return False

    def struct_nesting_depth(self):
        return 1

    def global_init_code(self, entry, code):
        pass

    def needs_nonecheck(self):
        return 0

    def _assign_from_py_code(self, source_code, result_code, error_pos, code, from_py_function=None, error_condition=None, extra_args=None, special_none_cvalue=None):
        args = ', ' + ', '.join(('%s' % arg for arg in extra_args)) if extra_args else ''
        convert_call = '%s(%s%s)' % (from_py_function or self.from_py_function, source_code, args)
        if self.is_enum:
            convert_call = typecast(self, c_long_type, convert_call)
        if special_none_cvalue:
            convert_call = '(__Pyx_Py_IsNone(%s) ? (%s) : (%s))' % (source_code, special_none_cvalue, convert_call)
        return '%s = %s; %s' % (result_code, convert_call, code.error_goto_if(error_condition or self.error_condition(result_code), error_pos))

    def _generate_dummy_refcounting(self, code, *ignored_args, **ignored_kwds):
        if self.needs_refcounting:
            raise NotImplementedError('Ref-counting operation not yet implemented for type %s' % self)

    def _generate_dummy_refcounting_assignment(self, code, cname, rhs_cname, *ignored_args, **ignored_kwds):
        if self.needs_refcounting:
            raise NotImplementedError('Ref-counting operation not yet implemented for type %s' % self)
        code.putln('%s = %s' % (cname, rhs_cname))
    generate_incref = generate_xincref = generate_decref = generate_xdecref = generate_decref_clear = generate_xdecref_clear = generate_gotref = generate_xgotref = generate_giveref = generate_xgiveref = _generate_dummy_refcounting
    generate_decref_set = generate_xdecref_set = _generate_dummy_refcounting_assignment

    def nullcheck_string(self, code, cname):
        if self.needs_refcounting:
            raise NotImplementedError('Ref-counting operation not yet implemented for type %s' % self)
        code.putln('1')

    def cpp_optional_declaration_code(self, entity_code, dll_linkage=None):
        raise NotImplementedError('cpp_optional_declaration_code only implemented for c++ classes and not type %s' % self)