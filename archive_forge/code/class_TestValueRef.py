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
class TestValueRef(BaseTest):

    def test_str(self):
        mod = self.module()
        glob = mod.get_global_variable('glob')
        self.assertEqual(str(glob), '@glob = global i32 0')

    def test_name(self):
        mod = self.module()
        glob = mod.get_global_variable('glob')
        self.assertEqual(glob.name, 'glob')
        glob.name = 'foobar'
        self.assertEqual(glob.name, 'foobar')

    def test_linkage(self):
        mod = self.module()
        glob = mod.get_global_variable('glob')
        linkage = glob.linkage
        self.assertIsInstance(glob.linkage, llvm.Linkage)
        glob.linkage = linkage
        self.assertEqual(glob.linkage, linkage)
        for linkage in ('internal', 'external'):
            glob.linkage = linkage
            self.assertIsInstance(glob.linkage, llvm.Linkage)
            self.assertEqual(glob.linkage.name, linkage)

    def test_visibility(self):
        mod = self.module()
        glob = mod.get_global_variable('glob')
        visibility = glob.visibility
        self.assertIsInstance(glob.visibility, llvm.Visibility)
        glob.visibility = visibility
        self.assertEqual(glob.visibility, visibility)
        for visibility in ('hidden', 'protected', 'default'):
            glob.visibility = visibility
            self.assertIsInstance(glob.visibility, llvm.Visibility)
            self.assertEqual(glob.visibility.name, visibility)

    def test_storage_class(self):
        mod = self.module()
        glob = mod.get_global_variable('glob')
        storage_class = glob.storage_class
        self.assertIsInstance(glob.storage_class, llvm.StorageClass)
        glob.storage_class = storage_class
        self.assertEqual(glob.storage_class, storage_class)
        for storage_class in ('dllimport', 'dllexport', 'default'):
            glob.storage_class = storage_class
            self.assertIsInstance(glob.storage_class, llvm.StorageClass)
            self.assertEqual(glob.storage_class.name, storage_class)

    def test_add_function_attribute(self):
        mod = self.module()
        fn = mod.get_function('sum')
        fn.add_function_attribute('nocapture')
        with self.assertRaises(ValueError) as raises:
            fn.add_function_attribute('zext')
        self.assertEqual(str(raises.exception), "no such attribute 'zext'")

    def test_module(self):
        mod = self.module()
        glob = mod.get_global_variable('glob')
        self.assertIs(glob.module, mod)

    def test_type(self):
        mod = self.module()
        glob = mod.get_global_variable('glob')
        tp = glob.type
        self.assertIsInstance(tp, llvm.TypeRef)

    def test_type_name(self):
        mod = self.module()
        glob = mod.get_global_variable('glob')
        tp = glob.type
        self.assertEqual(tp.name, '')
        st = mod.get_global_variable('glob_struct')
        self.assertIsNotNone(re.match('struct\\.glob_type(\\.[\\d]+)?', st.type.element_type.name))

    def test_type_printing_variable(self):
        mod = self.module()
        glob = mod.get_global_variable('glob')
        tp = glob.type
        self.assertEqual(str(tp), 'i32*')

    def test_type_printing_function(self):
        mod = self.module()
        fn = mod.get_function('sum')
        self.assertEqual(str(fn.type), 'i32 (i32, i32)*')

    def test_type_printing_struct(self):
        mod = self.module()
        st = mod.get_global_variable('glob_struct')
        self.assertTrue(st.type.is_pointer)
        self.assertIsNotNone(re.match('%struct\\.glob_type(\\.[\\d]+)?\\*', str(st.type)))
        self.assertIsNotNone(re.match('%struct\\.glob_type(\\.[\\d]+)? = type { i64, \\[2 x i64\\] }', str(st.type.element_type)))

    def test_close(self):
        glob = self.glob()
        glob.close()
        glob.close()

    def test_is_declaration(self):
        defined = self.module().get_function('sum')
        declared = self.module(asm_sum_declare).get_function('sum')
        self.assertFalse(defined.is_declaration)
        self.assertTrue(declared.is_declaration)

    def test_module_global_variables(self):
        mod = self.module(asm_sum)
        gvars = list(mod.global_variables)
        self.assertEqual(len(gvars), 4)
        for v in gvars:
            self.assertTrue(v.is_global)

    def test_module_functions(self):
        mod = self.module()
        funcs = list(mod.functions)
        self.assertEqual(len(funcs), 1)
        func = funcs[0]
        self.assertTrue(func.is_function)
        self.assertEqual(func.name, 'sum')
        with self.assertRaises(ValueError):
            func.instructions
        with self.assertRaises(ValueError):
            func.operands
        with self.assertRaises(ValueError):
            func.opcode

    def test_function_arguments(self):
        mod = self.module()
        func = mod.get_function('sum')
        self.assertTrue(func.is_function)
        args = list(func.arguments)
        self.assertEqual(len(args), 2)
        self.assertTrue(args[0].is_argument)
        self.assertTrue(args[1].is_argument)
        self.assertEqual(args[0].name, '.1')
        self.assertEqual(str(args[0].type), 'i32')
        self.assertEqual(args[1].name, '.2')
        self.assertEqual(str(args[1].type), 'i32')
        with self.assertRaises(ValueError):
            args[0].blocks
        with self.assertRaises(ValueError):
            args[0].arguments

    def test_function_blocks(self):
        func = self.module().get_function('sum')
        blocks = list(func.blocks)
        self.assertEqual(len(blocks), 1)
        block = blocks[0]
        self.assertTrue(block.is_block)

    def test_block_instructions(self):
        func = self.module().get_function('sum')
        insts = list(list(func.blocks)[0].instructions)
        self.assertEqual(len(insts), 3)
        self.assertTrue(insts[0].is_instruction)
        self.assertTrue(insts[1].is_instruction)
        self.assertTrue(insts[2].is_instruction)
        self.assertEqual(insts[0].opcode, 'add')
        self.assertEqual(insts[1].opcode, 'add')
        self.assertEqual(insts[2].opcode, 'ret')

    def test_instruction_operands(self):
        func = self.module().get_function('sum')
        add = list(list(func.blocks)[0].instructions)[0]
        self.assertEqual(add.opcode, 'add')
        operands = list(add.operands)
        self.assertEqual(len(operands), 2)
        self.assertTrue(operands[0].is_operand)
        self.assertTrue(operands[1].is_operand)
        self.assertEqual(operands[0].name, '.1')
        self.assertEqual(str(operands[0].type), 'i32')
        self.assertEqual(operands[1].name, '.2')
        self.assertEqual(str(operands[1].type), 'i32')

    def test_function_attributes(self):
        mod = self.module(asm_attributes)
        for func in mod.functions:
            attrs = list(func.attributes)
            if func.name == 'a_readonly_func':
                self.assertEqual(attrs, [b'readonly'])
            elif func.name == 'a_arg0_return_func':
                self.assertEqual(attrs, [])
                args = list(func.arguments)
                self.assertEqual(list(args[0].attributes), [b'returned'])
                self.assertEqual(list(args[1].attributes), [])

    def test_value_kind(self):
        mod = self.module()
        self.assertEqual(mod.get_global_variable('glob').value_kind, llvm.ValueKind.global_variable)
        func = mod.get_function('sum')
        self.assertEqual(func.value_kind, llvm.ValueKind.function)
        block = list(func.blocks)[0]
        self.assertEqual(block.value_kind, llvm.ValueKind.basic_block)
        inst = list(block.instructions)[1]
        self.assertEqual(inst.value_kind, llvm.ValueKind.instruction)
        self.assertEqual(list(inst.operands)[0].value_kind, llvm.ValueKind.constant_int)
        self.assertEqual(list(inst.operands)[1].value_kind, llvm.ValueKind.instruction)
        iasm_func = self.module(asm_inlineasm).get_function('foo')
        iasm_inst = list(list(iasm_func.blocks)[0].instructions)[0]
        self.assertEqual(list(iasm_inst.operands)[0].value_kind, llvm.ValueKind.inline_asm)

    def test_is_constant(self):
        mod = self.module()
        self.assertTrue(mod.get_global_variable('glob').is_constant)
        constant_operands = 0
        for func in mod.functions:
            self.assertTrue(func.is_constant)
            for block in func.blocks:
                self.assertFalse(block.is_constant)
                for inst in block.instructions:
                    self.assertFalse(inst.is_constant)
                    for op in inst.operands:
                        if op.is_constant:
                            constant_operands += 1
        self.assertEqual(constant_operands, 1)

    def test_constant_int(self):
        mod = self.module()
        func = mod.get_function('sum')
        insts = list(list(func.blocks)[0].instructions)
        self.assertEqual(insts[1].opcode, 'add')
        operands = list(insts[1].operands)
        self.assertTrue(operands[0].is_constant)
        self.assertFalse(operands[1].is_constant)
        self.assertEqual(operands[0].get_constant_value(), 0)
        with self.assertRaises(ValueError):
            operands[1].get_constant_value()
        mod = self.module(asm_sum3)
        func = mod.get_function('sum')
        insts = list(list(func.blocks)[0].instructions)
        posint64 = list(insts[1].operands)[0]
        negint64 = list(insts[2].operands)[0]
        self.assertEqual(posint64.get_constant_value(), 5)
        self.assertEqual(negint64.get_constant_value(signed_int=True), -5)
        as_u64 = negint64.get_constant_value(signed_int=False)
        as_i64 = int.from_bytes(as_u64.to_bytes(8, 'little'), 'little', signed=True)
        self.assertEqual(as_i64, -5)

    def test_constant_fp(self):
        mod = self.module(asm_double_locale)
        func = mod.get_function('foo')
        insts = list(list(func.blocks)[0].instructions)
        self.assertEqual(len(insts), 2)
        self.assertEqual(insts[0].opcode, 'fadd')
        operands = list(insts[0].operands)
        self.assertTrue(operands[0].is_constant)
        self.assertAlmostEqual(operands[0].get_constant_value(), 0.0)
        self.assertTrue(operands[1].is_constant)
        self.assertAlmostEqual(operands[1].get_constant_value(), 3.14)
        mod = self.module(asm_double_inaccurate)
        func = mod.get_function('foo')
        inst = list(list(func.blocks)[0].instructions)[0]
        operands = list(inst.operands)
        with self.assertRaises(ValueError):
            operands[0].get_constant_value()
        self.assertAlmostEqual(operands[1].get_constant_value(round_fp=True), 0)

    def test_constant_as_string(self):
        mod = self.module(asm_null_constant)
        func = mod.get_function('bar')
        inst = list(list(func.blocks)[0].instructions)[0]
        arg = list(inst.operands)[0]
        self.assertTrue(arg.is_constant)
        self.assertEqual(arg.get_constant_value(), 'i64* null')

    def test_incoming_phi_blocks(self):
        mod = self.module(asm_phi_blocks)
        func = mod.get_function('foo')
        blocks = list(func.blocks)
        instructions = list(blocks[-1].instructions)
        self.assertTrue(instructions[0].is_instruction)
        self.assertEqual(instructions[0].opcode, 'phi')
        incoming_blocks = list(instructions[0].incoming_blocks)
        self.assertEqual(len(incoming_blocks), 2)
        self.assertTrue(incoming_blocks[0].is_block)
        self.assertTrue(incoming_blocks[1].is_block)
        self.assertEqual(incoming_blocks[0], blocks[-1])
        self.assertEqual(incoming_blocks[1], blocks[0])
        self.assertNotEqual(instructions[1].opcode, 'phi')
        with self.assertRaises(ValueError):
            instructions[1].incoming_blocks