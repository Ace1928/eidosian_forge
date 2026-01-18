from ctypes import *
import unittest, sys
from test import support
import _ctypes_test
class Structures(unittest.TestCase):

    def test_struct_by_value(self):

        class POINT(Structure):
            _fields_ = [('x', c_long), ('y', c_long)]

        class RECT(Structure):
            _fields_ = [('left', c_long), ('top', c_long), ('right', c_long), ('bottom', c_long)]
        dll = CDLL(_ctypes_test.__file__)
        pt = POINT(15, 25)
        left = c_long.in_dll(dll, 'left')
        top = c_long.in_dll(dll, 'top')
        right = c_long.in_dll(dll, 'right')
        bottom = c_long.in_dll(dll, 'bottom')
        rect = RECT(left, top, right, bottom)
        PointInRect = dll.PointInRect
        PointInRect.argtypes = [POINTER(RECT), POINT]
        self.assertEqual(1, PointInRect(byref(rect), pt))
        ReturnRect = dll.ReturnRect
        ReturnRect.argtypes = [c_int, RECT, POINTER(RECT), POINT, RECT, POINTER(RECT), POINT, RECT]
        ReturnRect.restype = RECT
        for i in range(4):
            ret = ReturnRect(i, rect, pointer(rect), pt, rect, byref(rect), pt, rect)
            self.assertEqual(ret.left, left.value)
            self.assertEqual(ret.right, right.value)
            self.assertEqual(ret.top, top.value)
            self.assertEqual(ret.bottom, bottom.value)
        from ctypes import _pointer_type_cache
        del _pointer_type_cache[RECT]