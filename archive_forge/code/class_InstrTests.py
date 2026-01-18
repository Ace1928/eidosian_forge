import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import opcode
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase
class InstrTests(TestCase):

    def test_constructor(self):
        with self.assertRaises(TypeError):
            Instr('NOP', lineno='x')
        with self.assertRaises(ValueError):
            Instr('NOP', lineno=0)
        with self.assertRaises(TypeError):
            Instr(1)
        with self.assertRaises(ValueError):
            Instr('xxx')

    def test_repr(self):
        r = repr(Instr('NOP', lineno=10))
        self.assertIn('NOP', r)
        self.assertIn('10', r)
        self.assertIn('lineno', r)
        r = repr(Instr('LOAD_FAST', '_x_', lineno=10))
        self.assertIn('LOAD_FAST', r)
        self.assertIn('lineno', r)
        self.assertIn('10', r)
        self.assertIn('arg', r)
        self.assertIn('_x_', r)

    def test_invalid_arg(self):
        label = Label()
        block = BasicBlock()
        self.assertRaises(ValueError, Instr, 'EXTENDED_ARG', 0)
        self.assertRaises(TypeError, Instr, 'JUMP_ABSOLUTE', 1)
        self.assertRaises(TypeError, Instr, 'JUMP_ABSOLUTE', 1.0)
        Instr('JUMP_ABSOLUTE', label)
        Instr('JUMP_ABSOLUTE', block)
        self.assertRaises(TypeError, Instr, 'LOAD_DEREF', 'x')
        Instr('LOAD_DEREF', CellVar('x'))
        Instr('LOAD_DEREF', FreeVar('x'))
        self.assertRaises(TypeError, Instr, 'LOAD_FAST', 1)
        Instr('LOAD_FAST', 'x')
        self.assertRaises(TypeError, Instr, 'LOAD_NAME', 1)
        Instr('LOAD_NAME', 'x')
        self.assertRaises(ValueError, Instr, 'LOAD_CONST')
        self.assertRaises(ValueError, Instr, 'LOAD_CONST', label)
        self.assertRaises(ValueError, Instr, 'LOAD_CONST', block)
        Instr('LOAD_CONST', 1.0)
        Instr('LOAD_CONST', object())
        self.assertRaises(TypeError, Instr, 'COMPARE_OP', 1)
        Instr('COMPARE_OP', Compare.EQ)
        self.assertRaises(ValueError, Instr, 'CALL_FUNCTION', -1)
        self.assertRaises(TypeError, Instr, 'CALL_FUNCTION', 3.0)
        Instr('CALL_FUNCTION', 3)
        self.assertRaises(ValueError, Instr, 'CALL_FUNCTION', 2147483647 + 1)
        instr = Instr('CALL_FUNCTION', 2147483647)
        self.assertEqual(instr.arg, 2147483647)
        self.assertRaises(ValueError, Instr, 'NOP', 0)
        Instr('NOP')

    def test_require_arg(self):
        i = Instr('CALL_FUNCTION', 3)
        self.assertTrue(i.require_arg())
        i = Instr('NOP')
        self.assertFalse(i.require_arg())

    def test_attr(self):
        instr = Instr('LOAD_CONST', 3, lineno=5)
        self.assertEqual(instr.name, 'LOAD_CONST')
        self.assertEqual(instr.opcode, 100)
        self.assertEqual(instr.arg, 3)
        self.assertEqual(instr.lineno, 5)
        self.assertRaises(ValueError, setattr, instr, 'lineno', 0)
        self.assertRaises(TypeError, setattr, instr, 'lineno', 1.0)
        self.assertRaises(TypeError, setattr, instr, 'name', 5)
        self.assertRaises(TypeError, setattr, instr, 'opcode', 1.0)
        self.assertRaises(ValueError, setattr, instr, 'opcode', -1)
        self.assertRaises(ValueError, setattr, instr, 'opcode', 255)
        instr.arg = -8
        instr.arg = object()
        self.assertRaises(AttributeError, delattr, instr, 'arg')
        instr = Instr('ROT_TWO')
        self.assertIs(instr.arg, UNSET)

    def test_modify_op(self):
        instr = Instr('LOAD_NAME', 'x')
        load_fast = opcode.opmap['LOAD_FAST']
        instr.opcode = load_fast
        self.assertEqual(instr.name, 'LOAD_FAST')
        self.assertEqual(instr.opcode, load_fast)

    def test_extended_arg(self):
        instr = Instr('LOAD_CONST', 305441741)
        self.assertEqual(instr.arg, 305441741)

    def test_slots(self):
        instr = Instr('NOP')
        with self.assertRaises(AttributeError):
            instr.myattr = 1

    def test_compare(self):
        instr = Instr('LOAD_CONST', 3, lineno=7)
        self.assertEqual(instr, Instr('LOAD_CONST', 3, lineno=7))
        self.assertNotEqual(instr, 1)
        self.assertNotEqual(instr, Instr('LOAD_CONST', 3))
        self.assertNotEqual(instr, Instr('LOAD_CONST', 3, lineno=6))
        self.assertNotEqual(instr, Instr('LOAD_FAST', 'x', lineno=7))
        self.assertNotEqual(instr, Instr('LOAD_CONST', 4, lineno=7))

    def test_has_jump(self):
        label = Label()
        jump = Instr('JUMP_ABSOLUTE', label)
        self.assertTrue(jump.has_jump())
        instr = Instr('LOAD_FAST', 'x')
        self.assertFalse(instr.has_jump())

    def test_is_cond_jump(self):
        label = Label()
        jump = Instr('POP_JUMP_IF_TRUE', label)
        self.assertTrue(jump.is_cond_jump())
        instr = Instr('LOAD_FAST', 'x')
        self.assertFalse(instr.is_cond_jump())

    def test_is_uncond_jump(self):
        label = Label()
        jump = Instr('JUMP_ABSOLUTE', label)
        self.assertTrue(jump.is_uncond_jump())
        instr = Instr('POP_JUMP_IF_TRUE', label)
        self.assertFalse(instr.is_uncond_jump())

    def test_const_key_not_equal(self):

        def check(value):
            self.assertEqual(Instr('LOAD_CONST', value), Instr('LOAD_CONST', value))

        def func():
            pass
        check(None)
        check(0)
        check(0.0)
        check(b'bytes')
        check('text')
        check(Ellipsis)
        check((1, 2, 3))
        check(frozenset({1, 2, 3}))
        check(func.__code__)
        check(object())

    def test_const_key_equal(self):
        neg_zero = -0.0
        pos_zero = +0.0
        self.assertNotEqual(Instr('LOAD_CONST', 0), Instr('LOAD_CONST', 0.0))
        self.assertNotEqual(Instr('LOAD_CONST', neg_zero), Instr('LOAD_CONST', pos_zero))
        self.assertNotEqual(Instr('LOAD_CONST', complex(neg_zero, 1.0)), Instr('LOAD_CONST', complex(pos_zero, 1.0)))
        self.assertNotEqual(Instr('LOAD_CONST', complex(1.0, neg_zero)), Instr('LOAD_CONST', complex(1.0, pos_zero)))
        self.assertNotEqual(Instr('LOAD_CONST', (0,)), Instr('LOAD_CONST', (0.0,)))
        nested_tuple1 = (0,)
        nested_tuple1 = (nested_tuple1,)
        nested_tuple2 = (0.0,)
        nested_tuple2 = (nested_tuple2,)
        self.assertNotEqual(Instr('LOAD_CONST', nested_tuple1), Instr('LOAD_CONST', nested_tuple2))
        self.assertNotEqual(Instr('LOAD_CONST', frozenset({0})), Instr('LOAD_CONST', frozenset({0.0})))

    def test_stack_effects(self):
        from _pydevd_frame_eval.vendored.bytecode.concrete import ConcreteInstr

        def check(instr):
            jump = instr.stack_effect(jump=True)
            no_jump = instr.stack_effect(jump=False)
            max_effect = instr.stack_effect(jump=None)
            self.assertEqual(instr.stack_effect(), max_effect)
            self.assertEqual(max_effect, max(jump, no_jump))
            if not instr.has_jump():
                self.assertEqual(jump, no_jump)
        for name, op in opcode.opmap.items():
            with self.subTest(name):
                if op < opcode.HAVE_ARGUMENT:
                    check(ConcreteInstr(name))
                else:
                    for arg in range(256):
                        check(ConcreteInstr(name, arg))
        for arg in (2 ** 31, 2 ** 32, 2 ** 63, 2 ** 64, -1):
            self.assertEqual(Instr('LOAD_CONST', arg).stack_effect(), 1)

    def test_code_object_containing_mutable_data(self):
        from _pydevd_frame_eval.vendored.bytecode import Bytecode, Instr
        from types import CodeType

        def f():

            def g():
                return 'value'
            return g
        f_code = Bytecode.from_code(f.__code__)
        instr_load_code = None
        mutable_datum = [4, 2]
        for each in f_code:
            if isinstance(each, Instr) and each.name == 'LOAD_CONST' and isinstance(each.arg, CodeType):
                instr_load_code = each
                break
        self.assertIsNotNone(instr_load_code)
        g_code = Bytecode.from_code(instr_load_code.arg)
        g_code[0].arg = mutable_datum
        instr_load_code.arg = g_code.to_code()
        f.__code__ = f_code.to_code()
        self.assertIs(f()(), mutable_datum)