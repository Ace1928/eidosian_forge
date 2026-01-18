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
class CPointerBaseType(CType):
    subtypes = ['base_type']

    def __init__(self, base_type):
        self.base_type = base_type
        if base_type.is_cv_qualified:
            base_type = base_type.cv_base_type
        for char_type in (c_char_type, c_uchar_type, c_schar_type):
            if base_type.same_as(char_type):
                self.is_string = 1
                break
        else:
            if base_type.same_as(c_py_unicode_type):
                self.is_pyunicode_ptr = 1
        if self.is_string and (not base_type.is_error):
            if base_type.signed == 2:
                self.to_py_function = '__Pyx_PyObject_FromCString'
                if self.is_ptr:
                    self.from_py_function = '__Pyx_PyObject_As%sSString'
            elif base_type.signed:
                self.to_py_function = '__Pyx_PyObject_FromString'
                if self.is_ptr:
                    self.from_py_function = '__Pyx_PyObject_As%sString'
            else:
                self.to_py_function = '__Pyx_PyObject_FromCString'
                if self.is_ptr:
                    self.from_py_function = '__Pyx_PyObject_As%sUString'
            if self.is_ptr:
                self.from_py_function %= '' if self.base_type.is_const else 'Writable'
            self.exception_value = 'NULL'
        elif self.is_pyunicode_ptr and (not base_type.is_error):
            self.to_py_function = '__Pyx_PyUnicode_FromUnicode'
            self.to_py_utility_code = UtilityCode.load_cached('pyunicode_from_unicode', 'StringTools.c')
            if self.is_ptr:
                self.from_py_function = '__Pyx_PyUnicode_AsUnicode'
            self.exception_value = 'NULL'

    def py_type_name(self):
        if self.is_string:
            return 'bytes'
        elif self.is_pyunicode_ptr:
            return 'unicode'
        else:
            return super(CPointerBaseType, self).py_type_name()

    def literal_code(self, value):
        if self.is_string:
            assert isinstance(value, str)
            return '"%s"' % StringEncoding.escape_byte_string(value)
        return str(value)