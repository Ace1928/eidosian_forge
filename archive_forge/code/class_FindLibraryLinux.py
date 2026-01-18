import unittest
import unittest.mock
import os.path
import sys
import test.support
from test.support import os_helper
from ctypes import *
from ctypes.util import find_library
@unittest.skipUnless(sys.platform.startswith('linux'), 'Test only valid for Linux')
class FindLibraryLinux(unittest.TestCase):

    def test_find_on_libpath(self):
        import subprocess
        import tempfile
        try:
            p = subprocess.Popen(['gcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            out, _ = p.communicate()
        except OSError:
            raise unittest.SkipTest('gcc, needed for test, not available')
        with tempfile.TemporaryDirectory() as d:
            srcname = os.path.join(d, 'dummy.c')
            libname = 'py_ctypes_test_dummy'
            dstname = os.path.join(d, 'lib%s.so' % libname)
            with open(srcname, 'wb') as f:
                pass
            self.assertTrue(os.path.exists(srcname))
            cmd = ['gcc', '-o', dstname, '--shared', '-Wl,-soname,lib%s.so' % libname, srcname]
            out = subprocess.check_output(cmd)
            self.assertTrue(os.path.exists(dstname))
            self.assertIsNone(find_library(libname))
            with os_helper.EnvironmentVarGuard() as env:
                KEY = 'LD_LIBRARY_PATH'
                if KEY not in env:
                    v = d
                else:
                    v = '%s:%s' % (env[KEY], d)
                env.set(KEY, v)
                self.assertEqual(find_library(libname), 'lib%s.so' % libname)

    def test_find_library_with_gcc(self):
        with unittest.mock.patch('ctypes.util._findSoname_ldconfig', lambda *args: None):
            self.assertNotEqual(find_library('c'), None)

    def test_find_library_with_ld(self):
        with unittest.mock.patch('ctypes.util._findSoname_ldconfig', lambda *args: None), unittest.mock.patch('ctypes.util._findLib_gcc', lambda *args: None):
            self.assertNotEqual(find_library('c'), None)

    def test_gh114257(self):
        self.assertIsNone(find_library('libc'))