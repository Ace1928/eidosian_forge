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
class CNumericType(CType):
    is_numeric = 1
    default_value = '0'
    has_attributes = True
    scope = None
    sign_words = ('unsigned ', '', 'signed ')

    def __init__(self, rank, signed=1):
        self.rank = rank
        if rank > 0 and signed == SIGNED:
            signed = 1
        self.signed = signed

    def sign_and_name(self):
        s = self.sign_words[self.signed]
        n = rank_to_type_name[self.rank]
        return s + n

    def is_simple_buffer_dtype(self):
        return True

    def __repr__(self):
        return '<CNumericType %s>' % self.sign_and_name()

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        type_name = self.sign_and_name()
        if pyrex or for_display:
            base_code = type_name.replace('PY_LONG_LONG', 'long long')
        else:
            base_code = public_decl(type_name, dll_linkage)
        base_code = StringEncoding.EncodedString(base_code)
        return self.base_declaration_code(base_code, entity_code)

    def attributes_known(self):
        if self.scope is None:
            from . import Symtab
            self.scope = scope = Symtab.CClassScope('', None, visibility='extern', parent_type=self)
            scope.directives = {}
            scope.declare_cfunction('conjugate', CFuncType(self, [CFuncTypeArg('self', self, None)], nogil=True), pos=None, defining=1, cname=' ')
        return True

    def __lt__(self, other):
        """Sort based on rank, preferring signed over unsigned"""
        if other.is_numeric:
            return self.rank > other.rank and self.signed >= other.signed
        return True

    def py_type_name(self):
        if self.rank <= 4:
            return 'int'
        return 'float'