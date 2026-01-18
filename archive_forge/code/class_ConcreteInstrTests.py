import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import opcode
import sys
import textwrap
import types
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.concrete import OFFSET_AS_INSTRUCTION
from _pydevd_frame_eval.vendored.bytecode.tests import get_code, TestCase
class ConcreteInstrTests(TestCase):

    def test_constructor(self):
        with self.assertRaises(ValueError):
            ConcreteInstr('LOAD_CONST')
        with self.assertRaises(ValueError):
            ConcreteInstr('ROT_TWO', 33)
        with self.assertRaises(TypeError):
            ConcreteInstr('LOAD_CONST', 1.0)
        with self.assertRaises(ValueError):
            ConcreteInstr('LOAD_CONST', -1)
        with self.assertRaises(TypeError):
            ConcreteInstr('LOAD_CONST', 5, lineno=1.0)
        with self.assertRaises(ValueError):
            ConcreteInstr('LOAD_CONST', 5, lineno=-1)
        with self.assertRaises(ValueError):
            ConcreteInstr('LOAD_CONST', 2147483647 + 1)
        instr = ConcreteInstr('LOAD_CONST', 2147483647)
        self.assertEqual(instr.arg, 2147483647)
        instr = ConcreteInstr('LOAD_FAST', 8, lineno=3, extended_args=1)
        self.assertEqual(instr.name, 'LOAD_FAST')
        self.assertEqual(instr.arg, 8)
        self.assertEqual(instr.lineno, 3)
        self.assertEqual(instr.size, 4)

    def test_attr(self):
        instr = ConcreteInstr('LOAD_CONST', 5, lineno=12)
        self.assertEqual(instr.name, 'LOAD_CONST')
        self.assertEqual(instr.opcode, 100)
        self.assertEqual(instr.arg, 5)
        self.assertEqual(instr.lineno, 12)
        self.assertEqual(instr.size, 2)

    def test_set(self):
        instr = ConcreteInstr('LOAD_CONST', 5, lineno=3)
        instr.set('NOP')
        self.assertEqual(instr.name, 'NOP')
        self.assertIs(instr.arg, UNSET)
        self.assertEqual(instr.lineno, 3)
        instr.set('LOAD_FAST', 8)
        self.assertEqual(instr.name, 'LOAD_FAST')
        self.assertEqual(instr.arg, 8)
        self.assertEqual(instr.lineno, 3)
        with self.assertRaises(ValueError):
            instr.set('LOAD_CONST')
        with self.assertRaises(ValueError):
            instr.set('NOP', 5)

    def test_set_attr(self):
        instr = ConcreteInstr('LOAD_CONST', 5, lineno=12)
        instr.name = 'LOAD_FAST'
        self.assertEqual(instr.name, 'LOAD_FAST')
        self.assertEqual(instr.opcode, 124)
        self.assertRaises(TypeError, setattr, instr, 'name', 3)
        self.assertRaises(ValueError, setattr, instr, 'name', 'xxx')
        instr.opcode = 100
        self.assertEqual(instr.name, 'LOAD_CONST')
        self.assertEqual(instr.opcode, 100)
        self.assertRaises(ValueError, setattr, instr, 'opcode', -12)
        self.assertRaises(TypeError, setattr, instr, 'opcode', 'abc')
        instr.arg = 305441741
        self.assertEqual(instr.arg, 305441741)
        self.assertEqual(instr.size, 8)
        instr.arg = 0
        self.assertEqual(instr.arg, 0)
        self.assertEqual(instr.size, 2)
        self.assertRaises(ValueError, setattr, instr, 'arg', -1)
        self.assertRaises(ValueError, setattr, instr, 'arg', 2147483647 + 1)
        self.assertRaises(AttributeError, setattr, instr, 'size', 3)
        instr.lineno = 33
        self.assertEqual(instr.lineno, 33)
        self.assertRaises(TypeError, setattr, instr, 'lineno', 1.0)
        self.assertRaises(ValueError, setattr, instr, 'lineno', -1)

    def test_size(self):
        self.assertEqual(ConcreteInstr('ROT_TWO').size, 2)
        self.assertEqual(ConcreteInstr('LOAD_CONST', 3).size, 2)
        self.assertEqual(ConcreteInstr('LOAD_CONST', 305441741).size, 8)

    def test_disassemble(self):
        code = b'\t\x00d\x03'
        instr = ConcreteInstr.disassemble(1, code, 0)
        self.assertEqual(instr, ConcreteInstr('NOP', lineno=1))
        instr = ConcreteInstr.disassemble(2, code, 1 if OFFSET_AS_INSTRUCTION else 2)
        self.assertEqual(instr, ConcreteInstr('LOAD_CONST', 3, lineno=2))
        code = b'\x90\x12\x904\x90\xabd\xcd'
        instr = ConcreteInstr.disassemble(3, code, 0)
        self.assertEqual(instr, ConcreteInstr('EXTENDED_ARG', 18, lineno=3))

    def test_assemble(self):
        instr = ConcreteInstr('NOP')
        self.assertEqual(instr.assemble(), b'\t\x00')
        instr = ConcreteInstr('LOAD_CONST', 3)
        self.assertEqual(instr.assemble(), b'd\x03')
        instr = ConcreteInstr('LOAD_CONST', 305441741)
        self.assertEqual(instr.assemble(), b'\x90\x12\x904\x90\xabd\xcd')
        instr = ConcreteInstr('LOAD_CONST', 3, extended_args=1)
        self.assertEqual(instr.assemble(), b'\x90\x00d\x03')

    def test_get_jump_target(self):
        jump_abs = ConcreteInstr('JUMP_ABSOLUTE', 3)
        self.assertEqual(jump_abs.get_jump_target(100), 3)
        jump_forward = ConcreteInstr('JUMP_FORWARD', 5)
        self.assertEqual(jump_forward.get_jump_target(10), 16 if OFFSET_AS_INSTRUCTION else 17)