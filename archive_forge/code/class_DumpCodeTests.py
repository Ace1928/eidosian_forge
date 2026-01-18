import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import contextlib
import io
import sys
import textwrap
import unittest
from _pydevd_frame_eval.vendored import bytecode
from _pydevd_frame_eval.vendored.bytecode import Label, Instr, Bytecode, BasicBlock, ControlFlowGraph
from _pydevd_frame_eval.vendored.bytecode.concrete import OFFSET_AS_INSTRUCTION
from _pydevd_frame_eval.vendored.bytecode.tests import disassemble
class DumpCodeTests(unittest.TestCase):
    maxDiff = 80 * 100

    def check_dump_bytecode(self, code, expected, lineno=None):
        with contextlib.redirect_stdout(io.StringIO()) as stderr:
            if lineno is not None:
                bytecode.dump_bytecode(code, lineno=True)
            else:
                bytecode.dump_bytecode(code)
            output = stderr.getvalue()
        self.assertEqual(output, expected)

    def test_bytecode(self):
        source = '\n            def func(test):\n                if test == 1:\n                    return 1\n                elif test == 2:\n                    return 2\n                return 3\n        '
        code = disassemble(source, function=True)
        enum_repr = '<Compare.EQ: 2>'
        expected = f"\n    LOAD_FAST 'test'\n    LOAD_CONST 1\n    COMPARE_OP {enum_repr}\n    POP_JUMP_IF_FALSE <label_instr6>\n    LOAD_CONST 1\n    RETURN_VALUE\n\nlabel_instr6:\n    LOAD_FAST 'test'\n    LOAD_CONST 2\n    COMPARE_OP {enum_repr}\n    POP_JUMP_IF_FALSE <label_instr13>\n    LOAD_CONST 2\n    RETURN_VALUE\n\nlabel_instr13:\n    LOAD_CONST 3\n    RETURN_VALUE\n\n        "[1:].rstrip(' ')
        self.check_dump_bytecode(code, expected)
        expected = f"\n    L.  2   0: LOAD_FAST 'test'\n            1: LOAD_CONST 1\n            2: COMPARE_OP {enum_repr}\n            3: POP_JUMP_IF_FALSE <label_instr6>\n    L.  3   4: LOAD_CONST 1\n            5: RETURN_VALUE\n\nlabel_instr6:\n    L.  4   7: LOAD_FAST 'test'\n            8: LOAD_CONST 2\n            9: COMPARE_OP {enum_repr}\n           10: POP_JUMP_IF_FALSE <label_instr13>\n    L.  5  11: LOAD_CONST 2\n           12: RETURN_VALUE\n\nlabel_instr13:\n    L.  6  14: LOAD_CONST 3\n           15: RETURN_VALUE\n\n        "[1:].rstrip(' ')
        self.check_dump_bytecode(code, expected, lineno=True)

    def test_bytecode_broken_label(self):
        label = Label()
        code = Bytecode([Instr('JUMP_ABSOLUTE', label)])
        expected = '    JUMP_ABSOLUTE <error: unknown label>\n\n'
        self.check_dump_bytecode(code, expected)

    def test_blocks_broken_jump(self):
        block = BasicBlock()
        code = ControlFlowGraph()
        code[0].append(Instr('JUMP_ABSOLUTE', block))
        expected = textwrap.dedent('\n            block1:\n                JUMP_ABSOLUTE <error: unknown block>\n\n        ').lstrip('\n')
        self.check_dump_bytecode(code, expected)

    def test_bytecode_blocks(self):
        source = '\n            def func(test):\n                if test == 1:\n                    return 1\n                elif test == 2:\n                    return 2\n                return 3\n        '
        code = disassemble(source, function=True)
        code = ControlFlowGraph.from_bytecode(code)
        enum_repr = '<Compare.EQ: 2>'
        expected = textwrap.dedent(f"\n            block1:\n                LOAD_FAST 'test'\n                LOAD_CONST 1\n                COMPARE_OP {enum_repr}\n                POP_JUMP_IF_FALSE <block3>\n                -> block2\n\n            block2:\n                LOAD_CONST 1\n                RETURN_VALUE\n\n            block3:\n                LOAD_FAST 'test'\n                LOAD_CONST 2\n                COMPARE_OP {enum_repr}\n                POP_JUMP_IF_FALSE <block5>\n                -> block4\n\n            block4:\n                LOAD_CONST 2\n                RETURN_VALUE\n\n            block5:\n                LOAD_CONST 3\n                RETURN_VALUE\n\n        ").lstrip()
        self.check_dump_bytecode(code, expected)
        expected = textwrap.dedent(f"\n            block1:\n                L.  2   0: LOAD_FAST 'test'\n                        1: LOAD_CONST 1\n                        2: COMPARE_OP {enum_repr}\n                        3: POP_JUMP_IF_FALSE <block3>\n                -> block2\n\n            block2:\n                L.  3   0: LOAD_CONST 1\n                        1: RETURN_VALUE\n\n            block3:\n                L.  4   0: LOAD_FAST 'test'\n                        1: LOAD_CONST 2\n                        2: COMPARE_OP {enum_repr}\n                        3: POP_JUMP_IF_FALSE <block5>\n                -> block4\n\n            block4:\n                L.  5   0: LOAD_CONST 2\n                        1: RETURN_VALUE\n\n            block5:\n                L.  6   0: LOAD_CONST 3\n                        1: RETURN_VALUE\n\n        ").lstrip()
        self.check_dump_bytecode(code, expected, lineno=True)

    def test_concrete_bytecode(self):
        source = '\n            def func(test):\n                if test == 1:\n                    return 1\n                elif test == 2:\n                    return 2\n                return 3\n        '
        code = disassemble(source, function=True)
        code = code.to_concrete_bytecode()
        expected = f'\n  0    LOAD_FAST 0\n  2    LOAD_CONST 1\n  4    COMPARE_OP 2\n  6    POP_JUMP_IF_FALSE {(6 if OFFSET_AS_INSTRUCTION else 12)}\n  8    LOAD_CONST 1\n 10    RETURN_VALUE\n 12    LOAD_FAST 0\n 14    LOAD_CONST 2\n 16    COMPARE_OP 2\n 18    POP_JUMP_IF_FALSE {(12 if OFFSET_AS_INSTRUCTION else 24)}\n 20    LOAD_CONST 2\n 22    RETURN_VALUE\n 24    LOAD_CONST 3\n 26    RETURN_VALUE\n'.lstrip('\n')
        self.check_dump_bytecode(code, expected)
        expected = f'\nL.  2   0: LOAD_FAST 0\n        2: LOAD_CONST 1\n        4: COMPARE_OP 2\n        6: POP_JUMP_IF_FALSE {(6 if OFFSET_AS_INSTRUCTION else 12)}\nL.  3   8: LOAD_CONST 1\n       10: RETURN_VALUE\nL.  4  12: LOAD_FAST 0\n       14: LOAD_CONST 2\n       16: COMPARE_OP 2\n       18: POP_JUMP_IF_FALSE {(12 if OFFSET_AS_INSTRUCTION else 24)}\nL.  5  20: LOAD_CONST 2\n       22: RETURN_VALUE\nL.  6  24: LOAD_CONST 3\n       26: RETURN_VALUE\n'.lstrip('\n')
        self.check_dump_bytecode(code, expected, lineno=True)

    def test_type_validation(self):

        class T:
            first_lineno = 1
        with self.assertRaises(TypeError):
            bytecode.dump_bytecode(T())