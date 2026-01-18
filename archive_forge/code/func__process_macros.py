from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _process_macros(self, macros):
    for key, value in macros.items():
        value = value.strip()
        if _r_int_literal.match(value):
            self._add_integer_constant(key, value)
        elif value == '...':
            self._declare('macro ' + key, value)
        else:
            raise CDefError('only supports one of the following syntax:\n  #define %s ...     (literally dot-dot-dot)\n  #define %s NUMBER  (with NUMBER an integer constant, decimal/hex/octal)\ngot:\n  #define %s %s' % (key, key, key, value))