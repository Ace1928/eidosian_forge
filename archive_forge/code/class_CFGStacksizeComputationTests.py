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
class CFGStacksizeComputationTests(TestCase):

    def check_stack_size(self, func):
        code = func.__code__
        bytecode = Bytecode.from_code(code)
        cfg = ControlFlowGraph.from_bytecode(bytecode)
        self.assertEqual(code.co_stacksize, cfg.compute_stacksize())

    def test_empty_code(self):
        cfg = ControlFlowGraph()
        del cfg[0]
        self.assertEqual(cfg.compute_stacksize(), 0)

    def test_handling_of_set_lineno(self):
        code = Bytecode()
        code.first_lineno = 3
        code.extend([Instr('LOAD_CONST', 7), Instr('STORE_NAME', 'x'), SetLineno(4), Instr('LOAD_CONST', 8), Instr('STORE_NAME', 'y'), SetLineno(5), Instr('LOAD_CONST', 9), Instr('STORE_NAME', 'z')])
        self.assertEqual(code.compute_stacksize(), 1)

    def test_invalid_stacksize(self):
        code = Bytecode()
        code.extend([Instr('STORE_NAME', 'x')])
        with self.assertRaises(RuntimeError):
            code.compute_stacksize()

    def test_stack_size_computation_and(self):

        def test(arg1, *args, **kwargs):
            return arg1 and args
        self.check_stack_size(test)

    def test_stack_size_computation_or(self):

        def test(arg1, *args, **kwargs):
            return arg1 or args
        self.check_stack_size(test)

    def test_stack_size_computation_if_else(self):

        def test(arg1, *args, **kwargs):
            if args:
                return 0
            elif kwargs:
                return 1
            else:
                return 2
        self.check_stack_size(test)

    def test_stack_size_computation_for_loop_continue(self):

        def test(arg1, *args, **kwargs):
            for k in kwargs:
                if k in args:
                    continue
            else:
                return 1
        self.check_stack_size(test)

    def test_stack_size_computation_while_loop_break(self):

        def test(arg1, *args, **kwargs):
            while True:
                if arg1:
                    break
        self.check_stack_size(test)

    def test_stack_size_computation_with(self):

        def test(arg1, *args, **kwargs):
            with open(arg1) as f:
                return f.read()
        self.check_stack_size(test)

    def test_stack_size_computation_try_except(self):

        def test(arg1, *args, **kwargs):
            try:
                return args[0]
            except Exception:
                return 2
        self.check_stack_size(test)

    def test_stack_size_computation_try_finally(self):

        def test(arg1, *args, **kwargs):
            try:
                return args[0]
            finally:
                return 2
        self.check_stack_size(test)

    def test_stack_size_computation_try_except_finally(self):

        def test(arg1, *args, **kwargs):
            try:
                return args[0]
            except Exception:
                return 2
            finally:
                print('Interrupt')
        self.check_stack_size(test)

    def test_stack_size_computation_try_except_else_finally(self):

        def test(arg1, *args, **kwargs):
            try:
                return args[0]
            except Exception:
                return 2
            else:
                return arg1
            finally:
                print('Interrupt')
        self.check_stack_size(test)

    def test_stack_size_computation_nested_try_except_finally(self):

        def test(arg1, *args, **kwargs):
            k = 1
            try:
                getattr(arg1, k)
            except AttributeError:
                pass
            except Exception:
                try:
                    assert False
                except Exception:
                    return 2
                finally:
                    print('unexpected')
            finally:
                print('attempted to get {}'.format(k))
        self.check_stack_size(test)

    def test_stack_size_computation_nested_try_except_else_finally(self):

        def test(*args, **kwargs):
            try:
                v = args[1]
            except IndexError:
                try:
                    w = kwargs['value']
                except KeyError:
                    return -1
                else:
                    return w
                finally:
                    print('second finally')
            else:
                return v
            finally:
                print('first finally')
        cpython_stacksize = test.__code__.co_stacksize
        test.__code__ = Bytecode.from_code(test.__code__).to_code()
        self.assertLessEqual(test.__code__.co_stacksize, cpython_stacksize)
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(test(1, 4), 4)
            self.assertEqual(stdout.getvalue(), 'first finally\n')
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(test([], value=3), 3)
            self.assertEqual(stdout.getvalue(), 'second finally\nfirst finally\n')
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(test([], name=None), -1)
            self.assertEqual(stdout.getvalue(), 'second finally\nfirst finally\n')

    def test_stack_size_with_dead_code(self):

        def test(*args):
            return 0
            try:
                a = args[0]
            except IndexError:
                return -1
            else:
                return a
        test.__code__ = Bytecode.from_code(test.__code__).to_code()
        self.assertEqual(test.__code__.co_stacksize, 1)
        self.assertEqual(test(1), 0)

    def test_huge_code_with_numerous_blocks(self):

        def base_func(x):
            pass

        def mk_if_then_else(depth):
            instructions = []
            for i in range(depth):
                label_else = Label()
                instructions.extend([Instr('LOAD_FAST', 'x'), Instr('POP_JUMP_IF_FALSE', label_else), Instr('LOAD_GLOBAL', 'f{}'.format(i)), Instr('RETURN_VALUE'), label_else])
            instructions.extend([Instr('LOAD_CONST', None), Instr('RETURN_VALUE')])
            return instructions
        bytecode = Bytecode(mk_if_then_else(5000))
        bytecode.compute_stacksize()