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
class BytecodeToConcreteTests(TestCase):

    def test_label(self):
        code = Bytecode()
        label = Label()
        code.extend([Instr('LOAD_CONST', 'hello', lineno=1), Instr('JUMP_FORWARD', label, lineno=1), label, Instr('POP_TOP', lineno=1)])
        code = code.to_concrete_bytecode()
        expected = [ConcreteInstr('LOAD_CONST', 0, lineno=1), ConcreteInstr('JUMP_FORWARD', 0, lineno=1), ConcreteInstr('POP_TOP', lineno=1)]
        self.assertListEqual(list(code), expected)
        self.assertListEqual(code.consts, ['hello'])

    def test_label2(self):
        bytecode = Bytecode()
        label = Label()
        bytecode.extend([Instr('LOAD_NAME', 'test', lineno=1), Instr('POP_JUMP_IF_FALSE', label), Instr('LOAD_CONST', 5, lineno=2), Instr('STORE_NAME', 'x'), Instr('JUMP_FORWARD', label), Instr('LOAD_CONST', 7, lineno=4), Instr('STORE_NAME', 'x'), label, Instr('LOAD_CONST', None), Instr('RETURN_VALUE')])
        concrete = bytecode.to_concrete_bytecode()
        expected = [ConcreteInstr('LOAD_NAME', 0, lineno=1), ConcreteInstr('POP_JUMP_IF_FALSE', 7 if OFFSET_AS_INSTRUCTION else 14, lineno=1), ConcreteInstr('LOAD_CONST', 0, lineno=2), ConcreteInstr('STORE_NAME', 1, lineno=2), ConcreteInstr('JUMP_FORWARD', 2 if OFFSET_AS_INSTRUCTION else 4, lineno=2), ConcreteInstr('LOAD_CONST', 1, lineno=4), ConcreteInstr('STORE_NAME', 1, lineno=4), ConcreteInstr('LOAD_CONST', 2, lineno=4), ConcreteInstr('RETURN_VALUE', lineno=4)]
        self.assertListEqual(list(concrete), expected)
        self.assertListEqual(concrete.consts, [5, 7, None])
        self.assertListEqual(concrete.names, ['test', 'x'])
        self.assertListEqual(concrete.varnames, [])

    def test_label3(self):
        """
        CPython generates useless EXTENDED_ARG 0 in some cases. We need to
        properly track them as otherwise we can end up with broken offset for
        jumps.
        """
        source = '\n            def func(x):\n                if x == 1:\n                    return x + 0\n                elif x == 2:\n                    return x + 1\n                elif x == 3:\n                    return x + 2\n                elif x == 4:\n                    return x + 3\n                elif x == 5:\n                    return x + 4\n                elif x == 6:\n                    return x + 5\n                elif x == 7:\n                    return x + 6\n                elif x == 8:\n                    return x + 7\n                elif x == 9:\n                    return x + 8\n                elif x == 10:\n                    return x + 9\n                elif x == 11:\n                    return x + 10\n                elif x == 12:\n                    return x + 11\n                elif x == 13:\n                    return x + 12\n                elif x == 14:\n                    return x + 13\n                elif x == 15:\n                    return x + 14\n                elif x == 16:\n                    return x + 15\n                elif x == 17:\n                    return x + 16\n                return -1\n        '
        code = get_code(source, function=True)
        bcode = Bytecode.from_code(code)
        concrete = bcode.to_concrete_bytecode()
        self.assertIsInstance(concrete, ConcreteBytecode)
        loc = {}
        exec(textwrap.dedent(source), loc)
        func = loc['func']
        func.__code__ = bcode.to_code()
        for i, x in enumerate(range(1, 18)):
            self.assertEqual(func(x), x + i)
        self.assertEqual(func(18), -1)
        self.assertEqual(ConcreteBytecode.from_code(code).to_code().co_code, code.co_code)

    def test_setlineno(self):
        concrete = ConcreteBytecode()
        concrete.consts = [7, 8, 9]
        concrete.names = ['x', 'y', 'z']
        concrete.first_lineno = 3
        concrete.extend([ConcreteInstr('LOAD_CONST', 0), ConcreteInstr('STORE_NAME', 0), SetLineno(4), ConcreteInstr('LOAD_CONST', 1), ConcreteInstr('STORE_NAME', 1), SetLineno(5), ConcreteInstr('LOAD_CONST', 2), ConcreteInstr('STORE_NAME', 2)])
        code = concrete.to_bytecode()
        self.assertEqual(code, [Instr('LOAD_CONST', 7, lineno=3), Instr('STORE_NAME', 'x', lineno=3), Instr('LOAD_CONST', 8, lineno=4), Instr('STORE_NAME', 'y', lineno=4), Instr('LOAD_CONST', 9, lineno=5), Instr('STORE_NAME', 'z', lineno=5)])

    def test_extended_jump(self):
        NOP = bytes((opcode.opmap['NOP'],))

        class BigInstr(ConcreteInstr):

            def __init__(self, size):
                super().__init__('NOP')
                self._size = size

            def copy(self):
                return self

            def assemble(self):
                return NOP * self._size
        label = Label()
        nb_nop = 2 ** 16
        code = Bytecode([Instr('JUMP_ABSOLUTE', label), BigInstr(nb_nop), label, Instr('LOAD_CONST', None), Instr('RETURN_VALUE')])
        code_obj = code.to_code()
        if OFFSET_AS_INSTRUCTION:
            expected = b'\x90\x80q\x02' + NOP * nb_nop + b'd\x00S\x00'
        else:
            expected = b'\x90\x01\x90\x00q\x06' + NOP * nb_nop + b'd\x00S\x00'
        self.assertEqual(code_obj.co_code, expected)

    def test_jumps(self):
        code = Bytecode()
        label_else = Label()
        label_return = Label()
        code.extend([Instr('LOAD_NAME', 'test', lineno=1), Instr('POP_JUMP_IF_FALSE', label_else), Instr('LOAD_CONST', 12, lineno=2), Instr('STORE_NAME', 'x'), Instr('JUMP_FORWARD', label_return), label_else, Instr('LOAD_CONST', 37, lineno=4), Instr('STORE_NAME', 'x'), label_return, Instr('LOAD_CONST', None, lineno=4), Instr('RETURN_VALUE')])
        code = code.to_concrete_bytecode()
        expected = [ConcreteInstr('LOAD_NAME', 0, lineno=1), ConcreteInstr('POP_JUMP_IF_FALSE', 5 if OFFSET_AS_INSTRUCTION else 10, lineno=1), ConcreteInstr('LOAD_CONST', 0, lineno=2), ConcreteInstr('STORE_NAME', 1, lineno=2), ConcreteInstr('JUMP_FORWARD', 2 if OFFSET_AS_INSTRUCTION else 4, lineno=2), ConcreteInstr('LOAD_CONST', 1, lineno=4), ConcreteInstr('STORE_NAME', 1, lineno=4), ConcreteInstr('LOAD_CONST', 2, lineno=4), ConcreteInstr('RETURN_VALUE', lineno=4)]
        self.assertListEqual(list(code), expected)
        self.assertListEqual(code.consts, [12, 37, None])
        self.assertListEqual(code.names, ['test', 'x'])
        self.assertListEqual(code.varnames, [])

    def test_dont_merge_constants(self):
        code = Bytecode()
        code.extend([Instr('LOAD_CONST', 5, lineno=1), Instr('LOAD_CONST', 5.0, lineno=1), Instr('LOAD_CONST', -0.0, lineno=1), Instr('LOAD_CONST', +0.0, lineno=1)])
        code = code.to_concrete_bytecode()
        expected = [ConcreteInstr('LOAD_CONST', 0, lineno=1), ConcreteInstr('LOAD_CONST', 1, lineno=1), ConcreteInstr('LOAD_CONST', 2, lineno=1), ConcreteInstr('LOAD_CONST', 3, lineno=1)]
        self.assertListEqual(list(code), expected)
        self.assertListEqual(code.consts, [5, 5.0, -0.0, +0.0])

    def test_cellvars(self):
        code = Bytecode()
        code.cellvars = ['x']
        code.freevars = ['y']
        code.extend([Instr('LOAD_DEREF', CellVar('x'), lineno=1), Instr('LOAD_DEREF', FreeVar('y'), lineno=1)])
        concrete = code.to_concrete_bytecode()
        self.assertEqual(concrete.cellvars, ['x'])
        self.assertEqual(concrete.freevars, ['y'])
        code.extend([ConcreteInstr('LOAD_DEREF', 0, lineno=1), ConcreteInstr('LOAD_DEREF', 1, lineno=1)])

    def test_compute_jumps_convergence(self):
        code = Bytecode()
        label1 = Label()
        label2 = Label()
        nop = 'NOP'
        code.append(Instr('JUMP_ABSOLUTE', label1))
        code.append(Instr('JUMP_ABSOLUTE', label2))
        for x in range(4, 510 if OFFSET_AS_INSTRUCTION else 254, 2):
            code.append(Instr(nop))
        code.append(label1)
        code.append(Instr(nop))
        for x in range(514 if OFFSET_AS_INSTRUCTION else 256, 600 if OFFSET_AS_INSTRUCTION else 300, 2):
            code.append(Instr(nop))
        code.append(label2)
        code.append(Instr(nop))
        code.to_code()
        with self.assertRaises(RuntimeError):
            code.to_code(compute_jumps_passes=2)

    def test_extreme_compute_jumps_convergence(self):
        """Test of compute_jumps() requiring absurd number of passes.

        NOTE:  This test also serves to demonstrate that there is no worst
        case: the number of passes can be unlimited (or, actually, limited by
        the size of the provided code).

        This is an extension of test_compute_jumps_convergence.  Instead of
        two jumps, where the earlier gets extended after the latter, we
        instead generate a series of many jumps.  Each pass of compute_jumps()
        extends one more instruction, which in turn causes the one behind it
        to be extended on the next pass.

        """
        max_unextended_offset = 1 << 8
        unextended_branch_instr_size = 2
        N = max_unextended_offset // unextended_branch_instr_size
        if OFFSET_AS_INSTRUCTION:
            N *= 2
        nop = 'UNARY_POSITIVE'
        labels = [Label() for x in range(0, 3 * N)]
        code = Bytecode()
        code.extend((Instr('JUMP_FORWARD', labels[len(labels) - x - 1]) for x in range(0, len(labels))))
        end_of_jumps = len(code)
        code.extend((Instr(nop) for x in range(0, N)))
        offset = end_of_jumps + N
        for index in range(0, len(labels)):
            code.insert(offset, labels[index])
            if offset <= end_of_jumps:
                offset -= 1
            else:
                offset -= 2
        code.insert(0, Instr('LOAD_CONST', 0))
        del end_of_jumps
        code.append(Instr('RETURN_VALUE'))
        code.to_code(compute_jumps_passes=len(labels) + 1)

    def test_general_constants(self):
        """Test if general object could be linked as constants."""

        class CustomObject:
            pass

        class UnHashableCustomObject:
            __hash__ = None
        obj1 = [1, 2, 3]
        obj2 = {1, 2, 3}
        obj3 = CustomObject()
        obj4 = UnHashableCustomObject()
        code = Bytecode([Instr('LOAD_CONST', obj1, lineno=1), Instr('LOAD_CONST', obj2, lineno=1), Instr('LOAD_CONST', obj3, lineno=1), Instr('LOAD_CONST', obj4, lineno=1), Instr('BUILD_TUPLE', 4, lineno=1), Instr('RETURN_VALUE', lineno=1)])
        self.assertEqual(code.to_code().co_consts, (obj1, obj2, obj3, obj4))

        def f():
            return
        f.__code__ = code.to_code()
        self.assertEqual(f(), (obj1, obj2, obj3, obj4))