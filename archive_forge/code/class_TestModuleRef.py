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
class TestModuleRef(BaseTest):

    def test_str(self):
        mod = self.module()
        s = str(mod).strip()
        self.assertTrue(s.startswith('; ModuleID ='), s)

    def test_close(self):
        mod = self.module()
        str(mod)
        mod.close()
        with self.assertRaises(ctypes.ArgumentError):
            str(mod)
        mod.close()

    def test_with(self):
        mod = self.module()
        str(mod)
        with mod:
            str(mod)
        with self.assertRaises(ctypes.ArgumentError):
            str(mod)
        with self.assertRaises(RuntimeError):
            with mod:
                pass

    def test_name(self):
        mod = self.module()
        mod.name = 'foo'
        self.assertEqual(mod.name, 'foo')
        mod.name = 'bar'
        self.assertEqual(mod.name, 'bar')

    def test_source_file(self):
        mod = self.module()
        self.assertEqual(mod.source_file, 'asm_sum.c')

    def test_data_layout(self):
        mod = self.module()
        s = mod.data_layout
        self.assertIsInstance(s, str)
        mod.data_layout = s
        self.assertEqual(s, mod.data_layout)

    def test_triple(self):
        mod = self.module()
        s = mod.triple
        self.assertEqual(s, llvm.get_default_triple())
        mod.triple = ''
        self.assertEqual(mod.triple, '')

    def test_verify(self):
        mod = self.module()
        self.assertIs(mod.verify(), None)
        mod = self.module(asm_verification_fail)
        with self.assertRaises(RuntimeError) as cm:
            mod.verify()
        s = str(cm.exception)
        self.assertIn('%.bug = add i32 1, %.bug', s)

    def test_get_function(self):
        mod = self.module()
        fn = mod.get_function('sum')
        self.assertIsInstance(fn, llvm.ValueRef)
        self.assertEqual(fn.name, 'sum')
        with self.assertRaises(NameError):
            mod.get_function('foo')
        del mod
        str(fn.module)

    def test_get_struct_type(self):
        mod = self.module()
        st_ty = mod.get_struct_type('struct.glob_type')
        self.assertEqual(st_ty.name, 'struct.glob_type')
        self.assertIsNotNone(re.match('%struct\\.glob_type(\\.[\\d]+)? = type { i64, \\[2 x i64\\] }', str(st_ty)))
        with self.assertRaises(NameError):
            mod.get_struct_type('struct.doesnt_exist')

    def test_get_global_variable(self):
        mod = self.module()
        gv = mod.get_global_variable('glob')
        self.assertIsInstance(gv, llvm.ValueRef)
        self.assertEqual(gv.name, 'glob')
        with self.assertRaises(NameError):
            mod.get_global_variable('bar')
        del mod
        str(gv.module)

    def test_global_variables(self):
        mod = self.module()
        it = mod.global_variables
        del mod
        globs = sorted(it, key=lambda value: value.name)
        self.assertEqual(len(globs), 4)
        self.assertEqual([g.name for g in globs], ['glob', 'glob_b', 'glob_f', 'glob_struct'])

    def test_functions(self):
        mod = self.module()
        it = mod.functions
        del mod
        funcs = list(it)
        self.assertEqual(len(funcs), 1)
        self.assertEqual(funcs[0].name, 'sum')

    def test_structs(self):
        mod = self.module()
        it = mod.struct_types
        del mod
        structs = list(it)
        self.assertEqual(len(structs), 1)
        self.assertIsNotNone(re.match('struct\\.glob_type(\\.[\\d]+)?', structs[0].name))
        self.assertIsNotNone(re.match('%struct\\.glob_type(\\.[\\d]+)? = type { i64, \\[2 x i64\\] }', str(structs[0])))

    def test_link_in(self):
        dest = self.module()
        src = self.module(asm_mul)
        dest.link_in(src)
        self.assertEqual(sorted((f.name for f in dest.functions)), ['mul', 'sum'])
        dest.get_function('mul')
        dest.close()
        with self.assertRaises(ctypes.ArgumentError):
            src.get_function('mul')

    def test_link_in_preserve(self):
        dest = self.module()
        src2 = self.module(asm_mul)
        dest.link_in(src2, preserve=True)
        self.assertEqual(sorted((f.name for f in dest.functions)), ['mul', 'sum'])
        dest.close()
        self.assertEqual(sorted((f.name for f in src2.functions)), ['mul'])
        src2.get_function('mul')

    def test_link_in_error(self):
        dest = self.module()
        src = self.module(asm_sum2)
        with self.assertRaises(RuntimeError) as cm:
            dest.link_in(src)
        self.assertIn('symbol multiply defined', str(cm.exception))

    def test_as_bitcode(self):
        mod = self.module()
        bc = mod.as_bitcode()
        bitcode_wrapper_magic = b'\xde\xc0\x17\x0b'
        bitcode_magic = b'BC'
        self.assertTrue(bc.startswith(bitcode_magic) or bc.startswith(bitcode_wrapper_magic))

    def test_parse_bitcode_error(self):
        with self.assertRaises(RuntimeError) as cm:
            llvm.parse_bitcode(b'')
        self.assertIn('LLVM bitcode parsing error', str(cm.exception))
        if llvm.llvm_version_info[0] < 9:
            self.assertIn('Invalid bitcode signature', str(cm.exception))
        else:
            self.assertIn('file too small to contain bitcode header', str(cm.exception))

    def test_bitcode_roundtrip(self):
        context1 = llvm.create_context()
        bc = self.module(context=context1).as_bitcode()
        context2 = llvm.create_context()
        mod = llvm.parse_bitcode(bc, context2)
        self.assertEqual(mod.as_bitcode(), bc)
        mod.get_function('sum')
        mod.get_global_variable('glob')

    def test_cloning(self):
        m = self.module()
        cloned = m.clone()
        self.assertIsNot(cloned, m)
        self.assertEqual(cloned.as_bitcode(), m.as_bitcode())