import sys
from Cryptodome.Util.py3compat import tobytes, is_native_int
from Cryptodome.Util._raw_api import (backend, load_lib,
from ._IntegerBase import IntegerBase
class _MPZ(Structure):
    _fields_ = [('_mp_alloc', c_int), ('_mp_size', c_int), ('_mp_d', c_void_p)]