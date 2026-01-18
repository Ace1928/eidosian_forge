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
class TestTargetMachine(BaseTest):

    def test_add_analysis_passes(self):
        tm = self.target_machine(jit=False)
        pm = llvm.create_module_pass_manager()
        tm.add_analysis_passes(pm)

    def test_target_data_from_tm(self):
        tm = self.target_machine(jit=False)
        td = tm.target_data
        mod = self.module()
        gv_i32 = mod.get_global_variable('glob')
        pointer_size = 4 if sys.maxsize < 2 ** 32 else 8
        self.assertEqual(td.get_abi_size(gv_i32.type), pointer_size)