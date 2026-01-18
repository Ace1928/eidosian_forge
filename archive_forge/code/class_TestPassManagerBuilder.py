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
class TestPassManagerBuilder(BaseTest):

    def pmb(self):
        return llvm.PassManagerBuilder()

    def test_old_api(self):
        pmb = llvm.create_pass_manager_builder()
        pmb.inlining_threshold = 2
        pmb.opt_level = 3

    def test_close(self):
        pmb = self.pmb()
        pmb.close()
        pmb.close()

    def test_opt_level(self):
        pmb = self.pmb()
        self.assertIsInstance(pmb.opt_level, int)
        for i in range(4):
            pmb.opt_level = i
            self.assertEqual(pmb.opt_level, i)

    def test_size_level(self):
        pmb = self.pmb()
        self.assertIsInstance(pmb.size_level, int)
        for i in range(4):
            pmb.size_level = i
            self.assertEqual(pmb.size_level, i)

    def test_inlining_threshold(self):
        pmb = self.pmb()
        with self.assertRaises(NotImplementedError):
            pmb.inlining_threshold
        for i in (25, 80, 350):
            pmb.inlining_threshold = i

    def test_disable_unroll_loops(self):
        pmb = self.pmb()
        self.assertIsInstance(pmb.disable_unroll_loops, bool)
        for b in (True, False):
            pmb.disable_unroll_loops = b
            self.assertEqual(pmb.disable_unroll_loops, b)

    def test_loop_vectorize(self):
        pmb = self.pmb()
        self.assertIsInstance(pmb.loop_vectorize, bool)
        for b in (True, False):
            pmb.loop_vectorize = b
            self.assertEqual(pmb.loop_vectorize, b)

    def test_slp_vectorize(self):
        pmb = self.pmb()
        self.assertIsInstance(pmb.slp_vectorize, bool)
        for b in (True, False):
            pmb.slp_vectorize = b
            self.assertEqual(pmb.slp_vectorize, b)

    def test_populate_module_pass_manager(self):
        pmb = self.pmb()
        pm = llvm.create_module_pass_manager()
        pmb.populate(pm)
        pmb.close()
        pm.close()

    def test_populate_function_pass_manager(self):
        mod = self.module()
        pmb = self.pmb()
        pm = llvm.create_function_pass_manager(mod)
        pmb.populate(pm)
        pmb.close()
        pm.close()