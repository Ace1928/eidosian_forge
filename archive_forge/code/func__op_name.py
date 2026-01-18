import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def _op_name(a):
    if isinstance(a, z3.FuncDeclRef):
        f = a
    else:
        f = a.decl()
    k = f.kind()
    n = _z3_op_to_str.get(k, None)
    if n is None:
        return f.name()
    else:
        return n