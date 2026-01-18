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
class TestRISCVABI(BaseTest):
    """
    Test calling convention of floating point arguments of RISC-V
    using different ABI.
    """
    triple = 'riscv32-unknown-linux'

    def setUp(self):
        super().setUp()
        llvm.initialize_all_targets()
        llvm.initialize_all_asmprinters()

    def check_riscv_target(self):
        try:
            llvm.Target.from_triple(self.triple)
        except RuntimeError as e:
            if 'No available targets are compatible with triple' in str(e):
                self.skipTest('RISCV target unsupported by linked LLVM.')
            else:
                raise e

    def riscv_target_machine(self, **kwarg):
        lltarget = llvm.Target.from_triple(self.triple)
        return lltarget.create_target_machine(**kwarg)

    def fpadd_ll_module(self):
        f64 = ir.DoubleType()
        f32 = ir.FloatType()
        fnty = ir.FunctionType(f64, (f32, f64))
        module = ir.Module()
        func = ir.Function(module, fnty, name='fpadd')
        block = func.append_basic_block()
        builder = ir.IRBuilder(block)
        a, b = func.args
        arg0 = builder.fpext(a, f64)
        result = builder.fadd(arg0, b)
        builder.ret(result)
        llmod = llvm.parse_assembly(str(module))
        llmod.verify()
        return llmod

    def break_up_asm(self, asm):
        asm_list = []
        for line in asm.splitlines():
            s_line = line.strip()
            if not (s_line.startswith('.') or s_line.startswith('fpadd') or s_line == ''):
                asm_list.append(s_line)
        return asm_list

    def test_rv32d_ilp32(self):
        self.check_riscv_target()
        llmod = self.fpadd_ll_module()
        target = self.riscv_target_machine(features='+f,+d')
        self.assertEqual(self.break_up_asm(target.emit_assembly(llmod)), riscv_asm_ilp32)

    def test_rv32d_ilp32f(self):
        self.check_riscv_target()
        llmod = self.fpadd_ll_module()
        target = self.riscv_target_machine(features='+f,+d', abiname='ilp32f')
        self.assertEqual(self.break_up_asm(target.emit_assembly(llmod)), riscv_asm_ilp32f)

    def test_rv32d_ilp32d(self):
        self.check_riscv_target()
        llmod = self.fpadd_ll_module()
        target = self.riscv_target_machine(features='+f,+d', abiname='ilp32d')
        self.assertEqual(self.break_up_asm(target.emit_assembly(llmod)), riscv_asm_ilp32d)