import ctypes
import threading
from ctypes import CFUNCTYPE, c_int, c_int32
from ctypes.util import find_library
import gc
import locale
import os
import platform
import re
import subprocess
import sys
import unittest
from contextlib import contextmanager
from tempfile import mkstemp
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.binding import ffi
from llvmlite.tests import TestCase
class TestDependencies(BaseTest):
    """
    Test DLL dependencies are within a certain expected set.
    """

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux-specific test')
    @unittest.skipUnless(os.environ.get('LLVMLITE_DIST_TEST'), 'Distribution-specific test')
    def test_linux(self):
        lib_path = ffi.lib._name
        env = os.environ.copy()
        env['LANG'] = 'C'
        p = subprocess.Popen(['objdump', '-p', lib_path], stdout=subprocess.PIPE, env=env)
        out, _ = p.communicate()
        self.assertEqual(0, p.returncode)
        lib_pat = re.compile('^([+-_a-zA-Z0-9]+)\\.so(?:\\.\\d+){0,3}$')
        deps = set()
        for line in out.decode().splitlines():
            parts = line.split()
            if parts and parts[0] == 'NEEDED':
                dep = parts[1]
                m = lib_pat.match(dep)
                if len(parts) != 2 or not m:
                    self.fail('invalid NEEDED line: %r' % (line,))
                deps.add(m.group(1))
        if 'libc' not in deps or 'libpthread' not in deps:
            self.fail('failed parsing dependencies? got %r' % (deps,))
        allowed = set(['librt', 'libdl', 'libpthread', 'libz', 'libm', 'libgcc_s', 'libc', 'ld-linux', 'ld64'])
        if platform.python_implementation() == 'PyPy':
            allowed.add('libtinfo')
        for dep in deps:
            if not dep.startswith('ld-linux-') and dep not in allowed:
                self.fail('unexpected dependency %r in %r' % (dep, deps))