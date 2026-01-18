from ctypes import *
import unittest
class PointerTestCase(unittest.TestCase):

    def test_p_cint(self):
        i = c_int(42)
        x = pointer(i)
        self.assertEqual(x._objects, {'1': i})