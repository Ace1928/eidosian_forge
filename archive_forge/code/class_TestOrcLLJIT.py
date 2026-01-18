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
@unittest.skipUnless(platform.machine().startswith('x86'), 'x86 only')
class TestOrcLLJIT(BaseTest):

    def jit(self, asm=asm_sum, func_name='sum', target_machine=None, add_process=False, func_type=CFUNCTYPE(c_int, c_int, c_int), suppress_errors=False):
        lljit = llvm.create_lljit_compiler(target_machine, use_jit_link=False, suppress_errors=suppress_errors)
        builder = llvm.JITLibraryBuilder()
        if add_process:
            builder.add_current_process()
        rt = builder.add_ir(asm.format(triple=llvm.get_default_triple())).export_symbol(func_name).link(lljit, func_name)
        cfptr = rt[func_name]
        self.assertTrue(cfptr)
        self.assertEqual(func_name, rt.name)
        return (lljit, rt, func_type(cfptr))

    def test_define_symbol(self):
        lljit = llvm.create_lljit_compiler()
        rt = llvm.JITLibraryBuilder().import_symbol('__xyzzy', 1234).export_symbol('__xyzzy').link(lljit, 'foo')
        self.assertEqual(rt['__xyzzy'], 1234)

    def test_lookup_undefined_symbol_fails(self):
        lljit = llvm.create_lljit_compiler()
        with self.assertRaisesRegex(RuntimeError, 'No such library'):
            lljit.lookup('foo', '__foobar')
        rt = llvm.JITLibraryBuilder().import_symbol('__xyzzy', 1234).export_symbol('__xyzzy').link(lljit, 'foo')
        self.assertNotEqual(rt['__xyzzy'], 0)
        with self.assertRaisesRegex(RuntimeError, 'Symbols not found.*__foobar'):
            lljit.lookup('foo', '__foobar')

    def test_jit_link(self):
        if sys.platform == 'win32':
            with self.assertRaisesRegex(RuntimeError, 'JITLink .* Windows'):
                llvm.create_lljit_compiler(use_jit_link=True)
        else:
            self.assertIsNotNone(llvm.create_lljit_compiler(use_jit_link=True))

    def test_run_code(self):
        lljit, rt, cfunc = self.jit()
        with lljit:
            res = cfunc(2, -5)
            self.assertEqual(-3, res)

    def test_close(self):
        lljit, rt, cfunc = self.jit()
        lljit.close()
        lljit.close()
        with self.assertRaises(AssertionError):
            lljit.lookup('foo', 'fn')

    def test_with(self):
        lljit, rt, cfunc = self.jit()
        with lljit:
            pass
        with self.assertRaises(RuntimeError):
            with lljit:
                pass
        with self.assertRaises(AssertionError):
            lljit.lookup('foo', 'fn')

    def test_add_ir_module(self):
        lljit, rt_sum, cfunc_sum = self.jit()
        rt_mul = llvm.JITLibraryBuilder().add_ir(asm_mul.format(triple=llvm.get_default_triple())).export_symbol('mul').link(lljit, 'mul')
        res = CFUNCTYPE(c_int, c_int, c_int)(rt_mul['mul'])(2, -5)
        self.assertEqual(-10, res)
        self.assertNotEqual(lljit.lookup('sum', 'sum')['sum'], 0)
        self.assertNotEqual(lljit.lookup('mul', 'mul')['mul'], 0)
        with self.assertRaises(RuntimeError):
            lljit.lookup('sum', 'mul')
        with self.assertRaises(RuntimeError):
            lljit.lookup('mul', 'sum')

    def test_remove_module(self):
        lljit, rt_sum, _ = self.jit()
        del rt_sum
        gc.collect()
        with self.assertRaises(RuntimeError):
            lljit.lookup('sum', 'sum')
        lljit.close()

    def test_lib_depends(self):
        lljit, rt_sum, cfunc_sum = self.jit()
        rt_mul = llvm.JITLibraryBuilder().add_ir(asm_square_sum.format(triple=llvm.get_default_triple())).export_symbol('square_sum').add_jit_library('sum').link(lljit, 'square_sum')
        res = CFUNCTYPE(c_int, c_int, c_int)(rt_mul['square_sum'])(2, -5)
        self.assertEqual(9, res)

    def test_target_data(self):
        lljit, rt, _ = self.jit()
        td = lljit.target_data
        self.assertIs(lljit.target_data, td)
        str(td)
        del lljit
        str(td)

    def test_global_ctors_dtors(self):
        shared_value = c_int32(0)
        lljit = llvm.create_lljit_compiler()
        builder = llvm.JITLibraryBuilder()
        rt = builder.add_ir(asm_ext_ctors.format(triple=llvm.get_default_triple())).import_symbol('A', ctypes.addressof(shared_value)).export_symbol('foo').link(lljit, 'foo')
        foo = rt['foo']
        self.assertTrue(foo)
        self.assertEqual(CFUNCTYPE(c_int)(foo)(), 12)
        del rt
        self.assertNotEqual(shared_value.value, 20)

    def test_lookup_current_process_symbol_fails(self):
        msg = 'Failed to materialize symbols:.*getversion'
        with self.assertRaisesRegex(RuntimeError, msg):
            self.jit(asm_getversion, 'getversion', suppress_errors=True)

    def test_lookup_current_process_symbol(self):
        self.jit(asm_getversion, 'getversion', None, True)

    def test_thread_safe(self):
        lljit = llvm.create_lljit_compiler()
        llvm_ir = asm_sum.format(triple=llvm.get_default_triple())

        def compile_many(i):

            def do_work():
                tracking = []
                for c in range(50):
                    tracking.append(llvm.JITLibraryBuilder().add_ir(llvm_ir).export_symbol('sum').link(lljit, f'sum_{i}_{c}'))
            return do_work
        ths = [threading.Thread(target=compile_many(i)) for i in range(os.cpu_count())]
        for th in ths:
            th.start()
        for th in ths:
            th.join()

    def test_add_object_file(self):
        target_machine = self.target_machine(jit=False)
        mod = self.module()
        lljit = llvm.create_lljit_compiler(target_machine)
        rt = llvm.JITLibraryBuilder().add_object_img(target_machine.emit_object(mod)).export_symbol('sum').link(lljit, 'sum')
        sum = CFUNCTYPE(c_int, c_int, c_int)(rt['sum'])
        self.assertEqual(sum(2, 3), 5)

    def test_add_object_file_from_filesystem(self):
        target_machine = self.target_machine(jit=False)
        mod = self.module()
        obj_bin = target_machine.emit_object(mod)
        temp_desc, temp_path = mkstemp()
        try:
            with os.fdopen(temp_desc, 'wb') as f:
                f.write(obj_bin)
            lljit = llvm.create_lljit_compiler(target_machine)
            rt = llvm.JITLibraryBuilder().add_object_file(temp_path).export_symbol('sum').link(lljit, 'sum')
            sum = CFUNCTYPE(c_int, c_int, c_int)(rt['sum'])
            self.assertEqual(sum(2, 3), 5)
        finally:
            os.unlink(temp_path)