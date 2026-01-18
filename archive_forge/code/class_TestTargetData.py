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
class TestTargetData(BaseTest):

    def target_data(self):
        return llvm.create_target_data('e-m:e-i64:64-f80:128-n8:16:32:64-S128')

    def test_get_abi_size(self):
        td = self.target_data()
        glob = self.glob()
        self.assertEqual(td.get_abi_size(glob.type), 8)

    def test_get_pointee_abi_size(self):
        td = self.target_data()
        glob = self.glob()
        self.assertEqual(td.get_pointee_abi_size(glob.type), 4)
        glob = self.glob('glob_struct')
        self.assertEqual(td.get_pointee_abi_size(glob.type), 24)

    def test_get_struct_element_offset(self):
        td = self.target_data()
        glob = self.glob('glob_struct')
        with self.assertRaises(ValueError):
            td.get_element_offset(glob.type, 0)
        struct_type = glob.type.element_type
        self.assertEqual(td.get_element_offset(struct_type, 0), 0)
        self.assertEqual(td.get_element_offset(struct_type, 1), 8)