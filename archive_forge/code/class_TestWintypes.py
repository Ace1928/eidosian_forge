from ctypes import *
import unittest, sys
from test import support
import _ctypes_test
@unittest.skipUnless(sys.platform == 'win32', 'Windows-specific test')
class TestWintypes(unittest.TestCase):

    def test_HWND(self):
        from ctypes import wintypes
        self.assertEqual(sizeof(wintypes.HWND), sizeof(c_void_p))

    def test_PARAM(self):
        from ctypes import wintypes
        self.assertEqual(sizeof(wintypes.WPARAM), sizeof(c_void_p))
        self.assertEqual(sizeof(wintypes.LPARAM), sizeof(c_void_p))

    def test_COMError(self):
        from _ctypes import COMError
        if support.HAVE_DOCSTRINGS:
            self.assertEqual(COMError.__doc__, 'Raised when a COM method call failed.')
        ex = COMError(-1, 'text', ('details',))
        self.assertEqual(ex.hresult, -1)
        self.assertEqual(ex.text, 'text')
        self.assertEqual(ex.details, ('details',))