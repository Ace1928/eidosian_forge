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
class TestModulePassManager(BaseTest, PassManagerTestMixin):

    def pm(self):
        return llvm.create_module_pass_manager()

    def test_run(self):
        pm = self.pm()
        self.pmb().populate(pm)
        mod = self.module()
        orig_asm = str(mod)
        pm.run(mod)
        opt_asm = str(mod)
        opt_asm_split = opt_asm.splitlines()
        for idx, l in enumerate(opt_asm_split):
            if l.strip().startswith('ret i32'):
                toks = {'%.3', '%.4'}
                for t in toks:
                    if t in l:
                        break
                else:
                    raise RuntimeError('expected tokens not found')
                othertoken = (toks ^ {t}).pop()
                self.assertIn('%.3', orig_asm)
                self.assertNotIn(othertoken, opt_asm)
                break
        else:
            raise RuntimeError('expected IR not found')

    def test_run_with_remarks_successful_inline(self):
        pm = self.pm()
        pm.add_function_inlining_pass(70)
        self.pmb().populate(pm)
        mod = self.module(asm_inlineasm2)
        status, remarks = pm.run_with_remarks(mod)
        self.assertTrue(status)
        self.assertIn('Passed', remarks)
        self.assertIn('inlineme', remarks)

    def test_run_with_remarks_failed_inline(self):
        pm = self.pm()
        pm.add_function_inlining_pass(0)
        self.pmb().populate(pm)
        mod = self.module(asm_inlineasm3)
        status, remarks = pm.run_with_remarks(mod)
        self.assertTrue(status)
        self.assertIn('Missed', remarks)
        self.assertIn('inlineme', remarks)
        self.assertIn('noinline function attribute', remarks)

    def test_run_with_remarks_inline_filter_out(self):
        pm = self.pm()
        pm.add_function_inlining_pass(70)
        self.pmb().populate(pm)
        mod = self.module(asm_inlineasm2)
        status, remarks = pm.run_with_remarks(mod, remarks_filter='nothing')
        self.assertTrue(status)
        self.assertEqual('', remarks)

    def test_run_with_remarks_inline_filter_in(self):
        pm = self.pm()
        pm.add_function_inlining_pass(70)
        self.pmb().populate(pm)
        mod = self.module(asm_inlineasm2)
        status, remarks = pm.run_with_remarks(mod, remarks_filter='inlin.*')
        self.assertTrue(status)
        self.assertIn('Passed', remarks)
        self.assertIn('inlineme', remarks)