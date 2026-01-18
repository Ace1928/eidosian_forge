from ctypes import *
import unittest
class InitTest(unittest.TestCase):

    def test_get(self):
        y = Y()
        self.assertEqual((y.x.a, y.x.b), (0, 0))
        self.assertEqual(y.x.new_was_called, False)
        x = X()
        self.assertEqual((x.a, x.b), (9, 12))
        self.assertEqual(x.new_was_called, True)
        y.x = x
        self.assertEqual((y.x.a, y.x.b), (9, 12))
        self.assertEqual(y.x.new_was_called, False)