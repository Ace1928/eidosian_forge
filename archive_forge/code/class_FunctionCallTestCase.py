from ctypes import *
import unittest, sys
from test import support
import _ctypes_test
@unittest.skipUnless(sys.platform == 'win32', 'Windows-specific test')
class FunctionCallTestCase(unittest.TestCase):

    @unittest.skipUnless('MSC' in sys.version, 'SEH only supported by MSC')
    @unittest.skipIf(sys.executable.lower().endswith('_d.exe'), 'SEH not enabled in debug builds')
    def test_SEH(self):
        with support.disable_faulthandler():
            self.assertRaises(OSError, windll.kernel32.GetModuleHandleA, 32)

    def test_noargs(self):
        windll.user32.GetDesktopWindow()