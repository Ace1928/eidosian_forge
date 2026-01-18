from ctypes import *
import unittest
class ArrayTestCase(unittest.TestCase):

    def test_cint_array(self):
        INTARR = c_int * 3
        ia = INTARR()
        self.assertEqual(ia._objects, None)
        ia[0] = 1
        ia[1] = 2
        ia[2] = 3
        self.assertEqual(ia._objects, None)

        class X(Structure):
            _fields_ = [('x', c_int), ('a', INTARR)]
        x = X()
        x.x = 1000
        x.a[0] = 42
        x.a[1] = 96
        self.assertEqual(x._objects, None)
        x.a = ia
        self.assertEqual(x._objects, {'1': {}})