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
class TestFunctionPassManager(BaseTest, PassManagerTestMixin):

    def pm(self, mod=None):
        mod = mod or self.module()
        return llvm.create_function_pass_manager(mod)

    def test_initfini(self):
        pm = self.pm()
        pm.initialize()
        pm.finalize()

    def test_run(self):
        mod = self.module()
        fn = mod.get_function('sum')
        pm = self.pm(mod)
        self.pmb().populate(pm)
        mod.close()
        orig_asm = str(fn)
        pm.initialize()
        pm.run(fn)
        pm.finalize()
        opt_asm = str(fn)
        self.assertIn('%.4', orig_asm)
        self.assertNotIn('%.4', opt_asm)

    def test_run_with_remarks(self):
        mod = self.module(licm_asm)
        fn = mod.get_function('licm')
        pm = self.pm(mod)
        pm.add_licm_pass()
        self.pmb().populate(pm)
        mod.close()
        pm.initialize()
        ok, remarks = pm.run_with_remarks(fn)
        pm.finalize()
        self.assertTrue(ok)
        self.assertIn('Passed', remarks)
        self.assertIn('licm', remarks)

    def test_run_with_remarks_filter_out(self):
        mod = self.module(licm_asm)
        fn = mod.get_function('licm')
        pm = self.pm(mod)
        pm.add_licm_pass()
        self.pmb().populate(pm)
        mod.close()
        pm.initialize()
        ok, remarks = pm.run_with_remarks(fn, remarks_filter='nothing')
        pm.finalize()
        self.assertTrue(ok)
        self.assertEqual('', remarks)

    def test_run_with_remarks_filter_in(self):
        mod = self.module(licm_asm)
        fn = mod.get_function('licm')
        pm = self.pm(mod)
        pm.add_licm_pass()
        self.pmb().populate(pm)
        mod.close()
        pm.initialize()
        ok, remarks = pm.run_with_remarks(fn, remarks_filter='licm')
        pm.finalize()
        self.assertTrue(ok)
        self.assertIn('Passed', remarks)
        self.assertIn('licm', remarks)