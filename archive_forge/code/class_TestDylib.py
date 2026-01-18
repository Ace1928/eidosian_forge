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
class TestDylib(BaseTest):

    def test_bad_library(self):
        with self.assertRaises(RuntimeError):
            llvm.load_library_permanently('zzzasdkf;jasd;l')

    @unittest.skipUnless(platform.system() in ['Linux'], 'test only works on Linux')
    def test_libm(self):
        libm = find_library('m')
        llvm.load_library_permanently(libm)