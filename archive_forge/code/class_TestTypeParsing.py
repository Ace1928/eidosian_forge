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
class TestTypeParsing(BaseTest):

    @contextmanager
    def check_parsing(self):
        mod = ir.Module()
        yield mod
        asm = str(mod)
        llvm.parse_assembly(asm)

    def test_literal_struct(self):
        with self.check_parsing() as mod:
            typ = ir.LiteralStructType([ir.IntType(32)])
            gv = ir.GlobalVariable(mod, typ, 'foo')
            gv.initializer = ir.Constant(typ, [1])
        with self.check_parsing() as mod:
            typ = ir.LiteralStructType([ir.IntType(32)], packed=True)
            gv = ir.GlobalVariable(mod, typ, 'foo')
            gv.initializer = ir.Constant(typ, [1])