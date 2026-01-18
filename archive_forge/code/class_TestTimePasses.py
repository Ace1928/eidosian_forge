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
class TestTimePasses(BaseTest):

    def test_reporting(self):
        mp = llvm.create_module_pass_manager()
        pmb = llvm.create_pass_manager_builder()
        pmb.opt_level = 3
        pmb.populate(mp)
        try:
            llvm.set_time_passes(True)
            mp.run(self.module())
            mp.run(self.module())
            mp.run(self.module())
        finally:
            report = llvm.report_and_reset_timings()
            llvm.set_time_passes(False)
        self.assertIsInstance(report, str)
        self.assertEqual(report.count('Pass execution timing report'), 1)

    def test_empty_report(self):
        self.assertFalse(llvm.report_and_reset_timings())