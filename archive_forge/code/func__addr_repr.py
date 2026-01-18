import ctypes, ctypes.util, operator, sys
from . import model
def _addr_repr(self, address):
    if address == 0:
        return 'NULL'
    else:
        if address < 0:
            address += 1 << 8 * ctypes.sizeof(ctypes.c_void_p)
        return '0x%x' % address