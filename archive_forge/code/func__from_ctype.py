import ctypes
from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.core.singleton import S
from sympy.tensor.indexed import IndexedBase
from sympy.utilities.decorator import doctest_depends_on
def _from_ctype(self, ctype):
    if ctype == ctypes.c_int:
        return ll.IntType(32)
    if ctype == ctypes.c_double:
        return self.fp_type
    if ctype == ctypes.POINTER(ctypes.c_double):
        return ll.PointerType(self.fp_type)
    if ctype == ctypes.c_void_p:
        return ll.PointerType(ll.IntType(32))
    if ctype == ctypes.py_object:
        return ll.PointerType(ll.IntType(32))
    print('Unhandled ctype = %s' % str(ctype))