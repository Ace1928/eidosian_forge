import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import io
import sys
import unittest
import contextlib
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.concrete import OFFSET_AS_INSTRUCTION
from _pydevd_frame_eval.vendored.bytecode.tests import disassemble as _disassemble, TestCase
class BytecodeBlocksFunctionalTests(TestCase):

    def test_eq(self):
        source = 'x = 1 if test else 2'
        code1 = disassemble(source)
        code2 = disassemble(source)
        self.assertEqual(code1, code2)
        self.assertFalse(code1 == 1)
        cfg = ControlFlowGraph()
        cfg.argnames = 10
        self.assertFalse(code1 == cfg)
        cfg = ControlFlowGraph()
        cfg.argnames = code1.argnames
        self.assertFalse(code1 == cfg)

    def check_getitem(self, code):
        for block_index, block in enumerate(code):
            self.assertIs(code[block_index], block)
            self.assertIs(code[block], block)
            self.assertEqual(code.get_block_index(block), block_index)

    def test_delitem(self):
        cfg = ControlFlowGraph()
        b = cfg.add_block()
        del cfg[b]
        self.assertEqual(len(cfg.get_instructions()), 0)

    def sample_code(self):
        code = disassemble('x = 1', remove_last_return_none=True)
        self.assertBlocksEqual(code, [Instr('LOAD_CONST', 1, lineno=1), Instr('STORE_NAME', 'x', lineno=1)])
        return code

    def test_split_block(self):
        code = self.sample_code()
        code[0].append(Instr('NOP', lineno=1))
        label = code.split_block(code[0], 2)
        self.assertIs(label, code[1])
        self.assertBlocksEqual(code, [Instr('LOAD_CONST', 1, lineno=1), Instr('STORE_NAME', 'x', lineno=1)], [Instr('NOP', lineno=1)])
        self.check_getitem(code)
        label2 = code.split_block(code[0], 1)
        self.assertIs(label2, code[1])
        self.assertBlocksEqual(code, [Instr('LOAD_CONST', 1, lineno=1)], [Instr('STORE_NAME', 'x', lineno=1)], [Instr('NOP', lineno=1)])
        self.check_getitem(code)
        with self.assertRaises(TypeError):
            code.split_block(1, 1)
        with self.assertRaises(ValueError) as e:
            code.split_block(code[0], -2)
        self.assertIn('positive', e.exception.args[0])

    def test_split_block_end(self):
        code = self.sample_code()
        label = code.split_block(code[0], 2)
        self.assertIs(label, code[1])
        self.assertBlocksEqual(code, [Instr('LOAD_CONST', 1, lineno=1), Instr('STORE_NAME', 'x', lineno=1)], [])
        self.check_getitem(code)
        label = code.split_block(code[0], 2)
        self.assertIs(label, code[1])
        self.assertBlocksEqual(code, [Instr('LOAD_CONST', 1, lineno=1), Instr('STORE_NAME', 'x', lineno=1)], [])

    def test_split_block_dont_split(self):
        code = self.sample_code()
        block = code.split_block(code[0], 0)
        self.assertIs(block, code[0])
        self.assertBlocksEqual(code, [Instr('LOAD_CONST', 1, lineno=1), Instr('STORE_NAME', 'x', lineno=1)])

    def test_split_block_error(self):
        code = self.sample_code()
        with self.assertRaises(ValueError):
            code.split_block(code[0], 3)

    def test_to_code(self):
        bytecode = ControlFlowGraph()
        bytecode.first_lineno = 3
        bytecode.argcount = 3
        if sys.version_info > (3, 8):
            bytecode.posonlyargcount = 0
        bytecode.kwonlyargcount = 2
        bytecode.name = 'func'
        bytecode.filename = 'hello.py'
        bytecode.flags = 67
        bytecode.argnames = ('arg', 'arg2', 'arg3', 'kwonly', 'kwonly2')
        bytecode.docstring = None
        block0 = bytecode[0]
        block1 = bytecode.add_block()
        block2 = bytecode.add_block()
        block0.extend([Instr('LOAD_FAST', 'x', lineno=4), Instr('POP_JUMP_IF_FALSE', block2, lineno=4)])
        block1.extend([Instr('LOAD_FAST', 'arg', lineno=5), Instr('STORE_FAST', 'x', lineno=5)])
        block2.extend([Instr('LOAD_CONST', 3, lineno=6), Instr('STORE_FAST', 'x', lineno=6), Instr('LOAD_FAST', 'x', lineno=7), Instr('RETURN_VALUE', lineno=7)])
        if OFFSET_AS_INSTRUCTION:
            expected = b'|\x05r\x04|\x00}\x05d\x01}\x05|\x05S\x00'
        else:
            expected = b'|\x05r\x08|\x00}\x05d\x01}\x05|\x05S\x00'
        code = bytecode.to_code()
        self.assertEqual(code.co_consts, (None, 3))
        self.assertEqual(code.co_argcount, 3)
        if sys.version_info > (3, 8):
            self.assertEqual(code.co_posonlyargcount, 0)
        self.assertEqual(code.co_kwonlyargcount, 2)
        self.assertEqual(code.co_nlocals, 6)
        self.assertEqual(code.co_stacksize, 1)
        self.assertEqual(code.co_flags, 67)
        self.assertEqual(code.co_code, expected)
        self.assertEqual(code.co_names, ())
        self.assertEqual(code.co_varnames, ('arg', 'arg2', 'arg3', 'kwonly', 'kwonly2', 'x'))
        self.assertEqual(code.co_filename, 'hello.py')
        self.assertEqual(code.co_name, 'func')
        self.assertEqual(code.co_firstlineno, 3)
        explicit_stacksize = code.co_stacksize + 42
        code = bytecode.to_code(stacksize=explicit_stacksize)
        self.assertEqual(code.co_stacksize, explicit_stacksize)

    def test_get_block_index(self):
        blocks = ControlFlowGraph()
        block0 = blocks[0]
        block1 = blocks.add_block()
        block2 = blocks.add_block()
        self.assertEqual(blocks.get_block_index(block0), 0)
        self.assertEqual(blocks.get_block_index(block1), 1)
        self.assertEqual(blocks.get_block_index(block2), 2)
        other_block = BasicBlock()
        self.assertRaises(ValueError, blocks.get_block_index, other_block)