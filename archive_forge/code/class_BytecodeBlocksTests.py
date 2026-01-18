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
class BytecodeBlocksTests(TestCase):
    maxDiff = 80 * 100

    def test_constructor(self):
        code = ControlFlowGraph()
        self.assertEqual(code.name, '<module>')
        self.assertEqual(code.filename, '<string>')
        self.assertEqual(code.flags, 0)
        self.assertBlocksEqual(code, [])

    def test_attr(self):
        source = '\n            first_line = 1\n\n            def func(arg1, arg2, *, arg3):\n                x = 1\n                y = 2\n                return arg1\n        '
        code = disassemble(source, filename='hello.py', function=True)
        self.assertEqual(code.argcount, 2)
        self.assertEqual(code.filename, 'hello.py')
        self.assertEqual(code.first_lineno, 3)
        if sys.version_info > (3, 8):
            self.assertEqual(code.posonlyargcount, 0)
        self.assertEqual(code.kwonlyargcount, 1)
        self.assertEqual(code.name, 'func')
        self.assertEqual(code.cellvars, [])
        code.name = 'name'
        code.filename = 'filename'
        code.flags = 123
        self.assertEqual(code.name, 'name')
        self.assertEqual(code.filename, 'filename')
        self.assertEqual(code.flags, 123)

    def test_add_del_block(self):
        code = ControlFlowGraph()
        code[0].append(Instr('LOAD_CONST', 0))
        block = code.add_block()
        self.assertEqual(len(code), 2)
        self.assertIs(block, code[1])
        code[1].append(Instr('LOAD_CONST', 2))
        self.assertBlocksEqual(code, [Instr('LOAD_CONST', 0)], [Instr('LOAD_CONST', 2)])
        del code[0]
        self.assertBlocksEqual(code, [Instr('LOAD_CONST', 2)])
        del code[0]
        self.assertEqual(len(code), 0)

    def test_setlineno(self):
        code = Bytecode()
        code.first_lineno = 3
        code.extend([Instr('LOAD_CONST', 7), Instr('STORE_NAME', 'x'), SetLineno(4), Instr('LOAD_CONST', 8), Instr('STORE_NAME', 'y'), SetLineno(5), Instr('LOAD_CONST', 9), Instr('STORE_NAME', 'z')])
        blocks = ControlFlowGraph.from_bytecode(code)
        self.assertBlocksEqual(blocks, [Instr('LOAD_CONST', 7), Instr('STORE_NAME', 'x'), SetLineno(4), Instr('LOAD_CONST', 8), Instr('STORE_NAME', 'y'), SetLineno(5), Instr('LOAD_CONST', 9), Instr('STORE_NAME', 'z')])

    def test_legalize(self):
        code = Bytecode()
        code.first_lineno = 3
        code.extend([Instr('LOAD_CONST', 7), Instr('STORE_NAME', 'x'), Instr('LOAD_CONST', 8, lineno=4), Instr('STORE_NAME', 'y'), SetLineno(5), Instr('LOAD_CONST', 9, lineno=6), Instr('STORE_NAME', 'z')])
        blocks = ControlFlowGraph.from_bytecode(code)
        blocks.legalize()
        self.assertBlocksEqual(blocks, [Instr('LOAD_CONST', 7, lineno=3), Instr('STORE_NAME', 'x', lineno=3), Instr('LOAD_CONST', 8, lineno=4), Instr('STORE_NAME', 'y', lineno=4), Instr('LOAD_CONST', 9, lineno=5), Instr('STORE_NAME', 'z', lineno=5)])

    def test_repr(self):
        r = repr(ControlFlowGraph())
        self.assertIn('ControlFlowGraph', r)
        self.assertIn('1', r)

    def test_to_bytecode(self):
        blocks = ControlFlowGraph()
        blocks.add_block()
        blocks.add_block()
        blocks[0].extend([Instr('LOAD_NAME', 'test', lineno=1), Instr('POP_JUMP_IF_FALSE', blocks[2], lineno=1)])
        blocks[1].extend([Instr('LOAD_CONST', 5, lineno=2), Instr('STORE_NAME', 'x', lineno=2), Instr('JUMP_FORWARD', blocks[2], lineno=2)])
        blocks[2].extend([Instr('LOAD_CONST', 7, lineno=3), Instr('STORE_NAME', 'x', lineno=3), Instr('LOAD_CONST', None, lineno=3), Instr('RETURN_VALUE', lineno=3)])
        bytecode = blocks.to_bytecode()
        label = Label()
        self.assertEqual(bytecode, [Instr('LOAD_NAME', 'test', lineno=1), Instr('POP_JUMP_IF_FALSE', label, lineno=1), Instr('LOAD_CONST', 5, lineno=2), Instr('STORE_NAME', 'x', lineno=2), Instr('JUMP_FORWARD', label, lineno=2), label, Instr('LOAD_CONST', 7, lineno=3), Instr('STORE_NAME', 'x', lineno=3), Instr('LOAD_CONST', None, lineno=3), Instr('RETURN_VALUE', lineno=3)])

    def test_label_at_the_end(self):
        label = Label()
        code = Bytecode([Instr('LOAD_NAME', 'x'), Instr('UNARY_NOT'), Instr('POP_JUMP_IF_FALSE', label), Instr('LOAD_CONST', 9), Instr('STORE_NAME', 'y'), label])
        cfg = ControlFlowGraph.from_bytecode(code)
        self.assertBlocksEqual(cfg, [Instr('LOAD_NAME', 'x'), Instr('UNARY_NOT'), Instr('POP_JUMP_IF_FALSE', cfg[2])], [Instr('LOAD_CONST', 9), Instr('STORE_NAME', 'y')], [])

    def test_from_bytecode(self):
        bytecode = Bytecode()
        label = Label()
        bytecode.extend([Instr('LOAD_NAME', 'test', lineno=1), Instr('POP_JUMP_IF_FALSE', label, lineno=1), Instr('LOAD_CONST', 5, lineno=2), Instr('STORE_NAME', 'x', lineno=2), Instr('JUMP_FORWARD', label, lineno=2), Instr('LOAD_CONST', 7, lineno=4), Instr('STORE_NAME', 'x', lineno=4), Label(), label, Label(), Instr('LOAD_CONST', None, lineno=4), Instr('RETURN_VALUE', lineno=4)])
        blocks = ControlFlowGraph.from_bytecode(bytecode)
        label2 = blocks[3]
        self.assertBlocksEqual(blocks, [Instr('LOAD_NAME', 'test', lineno=1), Instr('POP_JUMP_IF_FALSE', label2, lineno=1)], [Instr('LOAD_CONST', 5, lineno=2), Instr('STORE_NAME', 'x', lineno=2), Instr('JUMP_FORWARD', label2, lineno=2)], [Instr('LOAD_CONST', 7, lineno=4), Instr('STORE_NAME', 'x', lineno=4)], [Instr('LOAD_CONST', None, lineno=4), Instr('RETURN_VALUE', lineno=4)])

    def test_from_bytecode_loop(self):
        if sys.version_info < (3, 8):
            label_loop_start = Label()
            label_loop_exit = Label()
            label_loop_end = Label()
            code = Bytecode()
            code.extend((Instr('SETUP_LOOP', label_loop_end, lineno=1), Instr('LOAD_CONST', (1, 2, 3), lineno=1), Instr('GET_ITER', lineno=1), label_loop_start, Instr('FOR_ITER', label_loop_exit, lineno=1), Instr('STORE_NAME', 'x', lineno=1), Instr('LOAD_NAME', 'x', lineno=2), Instr('LOAD_CONST', 2, lineno=2), Instr('COMPARE_OP', Compare.EQ, lineno=2), Instr('POP_JUMP_IF_FALSE', label_loop_start, lineno=2), Instr('BREAK_LOOP', lineno=3), Instr('JUMP_ABSOLUTE', label_loop_start, lineno=4), Instr('JUMP_ABSOLUTE', label_loop_start, lineno=4), label_loop_exit, Instr('POP_BLOCK', lineno=4), label_loop_end, Instr('LOAD_CONST', None, lineno=4), Instr('RETURN_VALUE', lineno=4)))
            blocks = ControlFlowGraph.from_bytecode(code)
            expected = [[Instr('SETUP_LOOP', blocks[8], lineno=1)], [Instr('LOAD_CONST', (1, 2, 3), lineno=1), Instr('GET_ITER', lineno=1)], [Instr('FOR_ITER', blocks[7], lineno=1)], [Instr('STORE_NAME', 'x', lineno=1), Instr('LOAD_NAME', 'x', lineno=2), Instr('LOAD_CONST', 2, lineno=2), Instr('COMPARE_OP', Compare.EQ, lineno=2), Instr('POP_JUMP_IF_FALSE', blocks[2], lineno=2)], [Instr('BREAK_LOOP', lineno=3)], [Instr('JUMP_ABSOLUTE', blocks[2], lineno=4)], [Instr('JUMP_ABSOLUTE', blocks[2], lineno=4)], [Instr('POP_BLOCK', lineno=4)], [Instr('LOAD_CONST', None, lineno=4), Instr('RETURN_VALUE', lineno=4)]]
            self.assertBlocksEqual(blocks, *expected)
        else:
            label_loop_start = Label()
            label_loop_exit = Label()
            code = Bytecode()
            code.extend((Instr('LOAD_CONST', (1, 2, 3), lineno=1), Instr('GET_ITER', lineno=1), label_loop_start, Instr('FOR_ITER', label_loop_exit, lineno=1), Instr('STORE_NAME', 'x', lineno=1), Instr('LOAD_NAME', 'x', lineno=2), Instr('LOAD_CONST', 2, lineno=2), Instr('COMPARE_OP', Compare.EQ, lineno=2), Instr('POP_JUMP_IF_FALSE', label_loop_start, lineno=2), Instr('JUMP_ABSOLUTE', label_loop_exit, lineno=3), Instr('JUMP_ABSOLUTE', label_loop_start, lineno=4), Instr('JUMP_ABSOLUTE', label_loop_start, lineno=4), label_loop_exit, Instr('LOAD_CONST', None, lineno=4), Instr('RETURN_VALUE', lineno=4)))
            blocks = ControlFlowGraph.from_bytecode(code)
            expected = [[Instr('LOAD_CONST', (1, 2, 3), lineno=1), Instr('GET_ITER', lineno=1)], [Instr('FOR_ITER', blocks[6], lineno=1)], [Instr('STORE_NAME', 'x', lineno=1), Instr('LOAD_NAME', 'x', lineno=2), Instr('LOAD_CONST', 2, lineno=2), Instr('COMPARE_OP', Compare.EQ, lineno=2), Instr('POP_JUMP_IF_FALSE', blocks[1], lineno=2)], [Instr('JUMP_ABSOLUTE', blocks[6], lineno=3)], [Instr('JUMP_ABSOLUTE', blocks[1], lineno=4)], [Instr('JUMP_ABSOLUTE', blocks[1], lineno=4)], [Instr('LOAD_CONST', None, lineno=4), Instr('RETURN_VALUE', lineno=4)]]
            self.assertBlocksEqual(blocks, *expected)