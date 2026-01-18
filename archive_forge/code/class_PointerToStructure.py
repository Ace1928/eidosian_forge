from ctypes import *
import unittest
class PointerToStructure(unittest.TestCase):

    def test(self):

        class POINT(Structure):
            _fields_ = [('x', c_int), ('y', c_int)]

        class RECT(Structure):
            _fields_ = [('a', POINTER(POINT)), ('b', POINTER(POINT))]
        r = RECT()
        p1 = POINT(1, 2)
        r.a = pointer(p1)
        r.b = pointer(p1)
        r.a[0].x = 42
        r.a[0].y = 99
        from ctypes import _pointer_type_cache
        del _pointer_type_cache[POINT]