from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _c_div(self, a, b):
    result = a // b
    if (a < 0) ^ (b < 0) and a % b != 0:
        result += 1
    return result