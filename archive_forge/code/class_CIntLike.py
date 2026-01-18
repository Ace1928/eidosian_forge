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
class CIntLike(object):
    """Mixin for shared behaviour of C integers and enums.
    """
    to_py_function = None
    from_py_function = None
    to_pyunicode_utility = None
    default_format_spec = 'd'

    def can_coerce_to_pyobject(self, env):
        return True

    def can_coerce_from_pyobject(self, env):
        return True

    def create_to_py_utility_code(self, env):
        if type(self).to_py_function is None:
            self.to_py_function = '__Pyx_PyInt_From_' + self.specialization_name()
            env.use_utility_code(TempitaUtilityCode.load_cached('CIntToPy', 'TypeConversion.c', context={'TYPE': self.empty_declaration_code(), 'TO_PY_FUNCTION': self.to_py_function}))
        return True

    def create_from_py_utility_code(self, env):
        if type(self).from_py_function is None:
            self.from_py_function = '__Pyx_PyInt_As_' + self.specialization_name()
            env.use_utility_code(TempitaUtilityCode.load_cached('CIntFromPy', 'TypeConversion.c', context={'TYPE': self.empty_declaration_code(), 'FROM_PY_FUNCTION': self.from_py_function, 'IS_ENUM': self.is_enum}))
        return True

    @staticmethod
    def _parse_format(format_spec):
        padding = ' '
        if not format_spec:
            return ('d', 0, padding)
        format_type = format_spec[-1]
        if format_type in ('o', 'd', 'x', 'X'):
            prefix = format_spec[:-1]
        elif format_type.isdigit():
            format_type = 'd'
            prefix = format_spec
        else:
            return (None, 0, padding)
        if not prefix:
            return (format_type, 0, padding)
        if prefix[0] == '-':
            prefix = prefix[1:]
        if prefix and prefix[0] == '0':
            padding = '0'
            prefix = prefix.lstrip('0')
        if prefix.isdigit():
            return (format_type, int(prefix), padding)
        return (None, 0, padding)

    def can_coerce_to_pystring(self, env, format_spec=None):
        format_type, width, padding = self._parse_format(format_spec)
        return format_type is not None and width <= 2 ** 30

    def convert_to_pystring(self, cvalue, code, format_spec=None):
        if self.to_pyunicode_utility is None:
            utility_code_name = '__Pyx_PyUnicode_From_' + self.specialization_name()
            to_pyunicode_utility = TempitaUtilityCode.load_cached('CIntToPyUnicode', 'TypeConversion.c', context={'TYPE': self.empty_declaration_code(), 'TO_PY_FUNCTION': utility_code_name})
            self.to_pyunicode_utility = (utility_code_name, to_pyunicode_utility)
        else:
            utility_code_name, to_pyunicode_utility = self.to_pyunicode_utility
        code.globalstate.use_utility_code(to_pyunicode_utility)
        format_type, width, padding_char = self._parse_format(format_spec)
        return "%s(%s, %d, '%s', '%s')" % (utility_code_name, cvalue, width, padding_char, format_type)