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
class CComplexType(CNumericType):
    is_complex = 1
    has_attributes = 1
    scope = None

    @property
    def to_py_function(self):
        return '__pyx_PyComplex_FromComplex%s' % self.implementation_suffix

    def __init__(self, real_type):
        while real_type.is_typedef and (not real_type.typedef_is_external):
            real_type = real_type.typedef_base_type
        self.funcsuffix = '_%s' % real_type.specialization_name()
        if not real_type.is_float:
            self.implementation_suffix = '_Cy'
        elif real_type.is_typedef and real_type.typedef_is_external:
            self.implementation_suffix = '_CyTypedef'
        else:
            self.implementation_suffix = ''
        if real_type.is_float:
            self.math_h_modifier = real_type.math_h_modifier
        else:
            self.math_h_modifier = '_UNUSED'
        self.real_type = real_type
        CNumericType.__init__(self, real_type.rank + 0.5, real_type.signed)
        self.binops = {}
        self.from_parts = '%s_from_parts' % self.specialization_name()
        self.default_value = '%s(0, 0)' % self.from_parts

    def __eq__(self, other):
        if isinstance(self, CComplexType) and isinstance(other, CComplexType):
            return self.real_type == other.real_type
        else:
            return False

    def __ne__(self, other):
        if isinstance(self, CComplexType) and isinstance(other, CComplexType):
            return self.real_type != other.real_type
        else:
            return True

    def __lt__(self, other):
        if isinstance(self, CComplexType) and isinstance(other, CComplexType):
            return self.real_type < other.real_type
        else:
            return False

    def __hash__(self):
        return ~hash(self.real_type)

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        if pyrex or for_display:
            real_code = self.real_type.declaration_code('', for_display, dll_linkage, pyrex)
            base_code = '%s complex' % real_code
        else:
            base_code = public_decl(self.sign_and_name(), dll_linkage)
        return self.base_declaration_code(base_code, entity_code)

    def sign_and_name(self):
        real_type_name = self.real_type.specialization_name()
        real_type_name = real_type_name.replace('long__double', 'long_double')
        real_type_name = real_type_name.replace('PY_LONG_LONG', 'long_long')
        return Naming.type_prefix + real_type_name + '_complex'

    def assignable_from(self, src_type):
        if not src_type.is_complex and src_type.is_numeric and src_type.is_typedef and src_type.typedef_is_external:
            return False
        elif src_type.is_pyobject:
            return True
        else:
            return super(CComplexType, self).assignable_from(src_type)

    def assignable_from_resolved_type(self, src_type):
        return src_type.is_complex and self.real_type.assignable_from_resolved_type(src_type.real_type) or (src_type.is_numeric and self.real_type.assignable_from_resolved_type(src_type)) or src_type is error_type

    def attributes_known(self):
        if self.scope is None:
            from . import Symtab
            self.scope = scope = Symtab.CClassScope('', None, visibility='extern', parent_type=self)
            scope.directives = {}
            scope.declare_var('real', self.real_type, None, cname='real', is_cdef=True)
            scope.declare_var('imag', self.real_type, None, cname='imag', is_cdef=True)
            scope.declare_cfunction('conjugate', CFuncType(self, [CFuncTypeArg('self', self, None)], nogil=True), pos=None, defining=1, cname='__Pyx_c_conj%s' % self.funcsuffix)
        return True

    def _utility_code_context(self):
        return {'type': self.empty_declaration_code(), 'type_name': self.specialization_name(), 'real_type': self.real_type.empty_declaration_code(), 'func_suffix': self.funcsuffix, 'm': self.math_h_modifier, 'is_float': int(self.real_type.is_float), 'is_extern_float_typedef': int(self.real_type.is_float and self.real_type.is_typedef and self.real_type.typedef_is_external)}

    def create_declaration_utility_code(self, env):
        if self.real_type.is_float:
            env.use_utility_code(UtilityCode.load_cached('Header', 'Complex.c'))
        utility_code_context = self._utility_code_context()
        env.use_utility_code(UtilityCode.load_cached('RealImag' + self.implementation_suffix, 'Complex.c'))
        env.use_utility_code(TempitaUtilityCode.load_cached('Declarations', 'Complex.c', utility_code_context))
        env.use_utility_code(TempitaUtilityCode.load_cached('Arithmetic', 'Complex.c', utility_code_context))
        return True

    def can_coerce_to_pyobject(self, env):
        return True

    def can_coerce_from_pyobject(self, env):
        return True

    def create_to_py_utility_code(self, env):
        env.use_utility_code(TempitaUtilityCode.load_cached('ToPy', 'Complex.c', self._utility_code_context()))
        return True

    def create_from_py_utility_code(self, env):
        env.use_utility_code(TempitaUtilityCode.load_cached('FromPy', 'Complex.c', self._utility_code_context()))
        self.from_py_function = '__Pyx_PyComplex_As_' + self.specialization_name()
        return True

    def lookup_op(self, nargs, op):
        try:
            return self.binops[nargs, op]
        except KeyError:
            pass
        try:
            op_name = complex_ops[nargs, op]
            self.binops[nargs, op] = func_name = '__Pyx_c_%s%s' % (op_name, self.funcsuffix)
            return func_name
        except KeyError:
            return None

    def unary_op(self, op):
        return self.lookup_op(1, op)

    def binary_op(self, op):
        return self.lookup_op(2, op)

    def py_type_name(self):
        return 'complex'

    def cast_code(self, expr_code):
        return expr_code

    def real_code(self, expr_code):
        return '__Pyx_CREAL%s(%s)' % (self.implementation_suffix, expr_code)

    def imag_code(self, expr_code):
        return '__Pyx_CIMAG%s(%s)' % (self.implementation_suffix, expr_code)