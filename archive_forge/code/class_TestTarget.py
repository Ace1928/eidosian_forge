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
class TestTarget(BaseTest):

    def test_from_triple(self):
        f = llvm.Target.from_triple
        with self.assertRaises(RuntimeError) as cm:
            f('foobar')
        self.assertIn('No available targets are compatible with', str(cm.exception))
        triple = llvm.get_default_triple()
        target = f(triple)
        self.assertEqual(target.triple, triple)
        target.close()

    def test_create_target_machine(self):
        target = llvm.Target.from_triple(llvm.get_default_triple())
        target.create_target_machine('', '', 1, 'default', 'default')
        cpu = llvm.get_host_cpu_name()
        target.create_target_machine(cpu, '', 1, 'default', 'default')

    def test_name(self):
        t = llvm.Target.from_triple(llvm.get_default_triple())
        u = llvm.Target.from_default_triple()
        self.assertIsInstance(t.name, str)
        self.assertEqual(t.name, u.name)

    def test_description(self):
        t = llvm.Target.from_triple(llvm.get_default_triple())
        u = llvm.Target.from_default_triple()
        self.assertIsInstance(t.description, str)
        self.assertEqual(t.description, u.description)

    def test_str(self):
        target = llvm.Target.from_triple(llvm.get_default_triple())
        s = str(target)
        self.assertIn(target.name, s)
        self.assertIn(target.description, s)