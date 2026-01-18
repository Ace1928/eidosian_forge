from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _add_integer_constant(self, name, int_str):
    int_str = int_str.lower().rstrip('ul')
    neg = int_str.startswith('-')
    if neg:
        int_str = int_str[1:]
    if int_str.startswith('0') and int_str != '0' and (not int_str.startswith('0x')):
        int_str = '0o' + int_str[1:]
    pyvalue = int(int_str, 0)
    if neg:
        pyvalue = -pyvalue
    self._add_constants(name, pyvalue)
    self._declare('macro ' + name, pyvalue)