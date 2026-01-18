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
class ConcreteFromCodeTests(TestCase):

    def test_extended_arg(self):
        co_code = b'\x90\x12\x904\x90\xabd\xcd'
        code = get_code('x=1')
        args = (code.co_argcount,) if sys.version_info < (3, 8) else (code.co_argcount, code.co_posonlyargcount)
        args += (code.co_kwonlyargcount, code.co_nlocals, code.co_stacksize, code.co_flags, co_code, code.co_consts, code.co_names, code.co_varnames, code.co_filename, code.co_name, code.co_firstlineno, code.co_linetable if sys.version_info >= (3, 10) else code.co_lnotab, code.co_freevars, code.co_cellvars)
        code = types.CodeType(*args)
        bytecode = ConcreteBytecode.from_code(code)
        self.assertListEqual(list(bytecode), [ConcreteInstr('LOAD_CONST', 305441741, lineno=1)])
        bytecode = ConcreteBytecode.from_code(code, extended_arg=True)
        expected = [ConcreteInstr('EXTENDED_ARG', 18, lineno=1), ConcreteInstr('EXTENDED_ARG', 52, lineno=1), ConcreteInstr('EXTENDED_ARG', 171, lineno=1), ConcreteInstr('LOAD_CONST', 205, lineno=1)]
        self.assertListEqual(list(bytecode), expected)

    def test_extended_arg_make_function(self):
        if (3, 9) <= sys.version_info < (3, 10):
            from _pydevd_frame_eval.vendored.bytecode.tests.util_annotation import get_code as get_code_future
            code_obj = get_code_future('\n                def foo(x: int, y: int):\n                    pass\n                ')
        else:
            code_obj = get_code('\n                def foo(x: int, y: int):\n                    pass\n                ')
        concrete = ConcreteBytecode.from_code(code_obj)
        if sys.version_info >= (3, 10):
            func_code = concrete.consts[2]
            names = ['int', 'foo']
            consts = ['x', 'y', func_code, 'foo', None]
            const_offset = 1
            name_offset = 1
            first_instrs = [ConcreteInstr('LOAD_CONST', 0, lineno=1), ConcreteInstr('LOAD_NAME', 0, lineno=1), ConcreteInstr('LOAD_CONST', 1, lineno=1), ConcreteInstr('LOAD_NAME', 0, lineno=1), ConcreteInstr('BUILD_TUPLE', 4, lineno=1)]
        elif sys.version_info >= (3, 7) and concrete.flags & CompilerFlags.FUTURE_ANNOTATIONS:
            func_code = concrete.consts[2]
            names = ['foo']
            consts = ['int', ('x', 'y'), func_code, 'foo', None]
            const_offset = 1
            name_offset = 0
            first_instrs = [ConcreteInstr('LOAD_CONST', 0, lineno=1), ConcreteInstr('LOAD_CONST', 0, lineno=1), ConcreteInstr('LOAD_CONST', 0 + const_offset, lineno=1), ConcreteInstr('BUILD_CONST_KEY_MAP', 2, lineno=1)]
        else:
            func_code = concrete.consts[1]
            names = ['int', 'foo']
            consts = [('x', 'y'), func_code, 'foo', None]
            const_offset = 0
            name_offset = 1
            first_instrs = [ConcreteInstr('LOAD_NAME', 0, lineno=1), ConcreteInstr('LOAD_NAME', 0, lineno=1), ConcreteInstr('LOAD_CONST', 0 + const_offset, lineno=1), ConcreteInstr('BUILD_CONST_KEY_MAP', 2, lineno=1)]
        self.assertEqual(concrete.names, names)
        self.assertEqual(concrete.consts, consts)
        expected = first_instrs + [ConcreteInstr('LOAD_CONST', 1 + const_offset, lineno=1), ConcreteInstr('LOAD_CONST', 2 + const_offset, lineno=1), ConcreteInstr('MAKE_FUNCTION', 4, lineno=1), ConcreteInstr('STORE_NAME', name_offset, lineno=1), ConcreteInstr('LOAD_CONST', 3 + const_offset, lineno=1), ConcreteInstr('RETURN_VALUE', lineno=1)]
        self.assertListEqual(list(concrete), expected)
        concrete = ConcreteBytecode.from_code(code_obj, extended_arg=True)
        if sys.version_info >= (3, 10):
            func_code = concrete.consts[2]
            names = ['int', 'foo']
            consts = ['x', 'y', func_code, 'foo', None]
        elif concrete.flags & CompilerFlags.FUTURE_ANNOTATIONS:
            func_code = concrete.consts[2]
            names = ['foo']
            consts = ['int', ('x', 'y'), func_code, 'foo', None]
        else:
            func_code = concrete.consts[1]
            names = ['int', 'foo']
            consts = [('x', 'y'), func_code, 'foo', None]
        self.assertEqual(concrete.names, names)
        self.assertEqual(concrete.consts, consts)
        self.assertListEqual(list(concrete), expected)

    def test_extended_arg_unpack_ex(self):

        def test():
            p = [1, 2, 3, 4, 5, 6]
            q, r, *s, t = p
            return (q, r, s, t)
        cpython_stacksize = test.__code__.co_stacksize
        test.__code__ = ConcreteBytecode.from_code(test.__code__, extended_arg=True).to_code()
        self.assertEqual(test.__code__.co_stacksize, cpython_stacksize)
        self.assertEqual(test(), (1, 2, [3, 4, 5], 6))

    def test_expected_arg_with_many_consts(self):

        def test():
            var = 0
            var = 1
            var = 2
            var = 3
            var = 4
            var = 5
            var = 6
            var = 7
            var = 8
            var = 9
            var = 10
            var = 11
            var = 12
            var = 13
            var = 14
            var = 15
            var = 16
            var = 17
            var = 18
            var = 19
            var = 20
            var = 21
            var = 22
            var = 23
            var = 24
            var = 25
            var = 26
            var = 27
            var = 28
            var = 29
            var = 30
            var = 31
            var = 32
            var = 33
            var = 34
            var = 35
            var = 36
            var = 37
            var = 38
            var = 39
            var = 40
            var = 41
            var = 42
            var = 43
            var = 44
            var = 45
            var = 46
            var = 47
            var = 48
            var = 49
            var = 50
            var = 51
            var = 52
            var = 53
            var = 54
            var = 55
            var = 56
            var = 57
            var = 58
            var = 59
            var = 60
            var = 61
            var = 62
            var = 63
            var = 64
            var = 65
            var = 66
            var = 67
            var = 68
            var = 69
            var = 70
            var = 71
            var = 72
            var = 73
            var = 74
            var = 75
            var = 76
            var = 77
            var = 78
            var = 79
            var = 80
            var = 81
            var = 82
            var = 83
            var = 84
            var = 85
            var = 86
            var = 87
            var = 88
            var = 89
            var = 90
            var = 91
            var = 92
            var = 93
            var = 94
            var = 95
            var = 96
            var = 97
            var = 98
            var = 99
            var = 100
            var = 101
            var = 102
            var = 103
            var = 104
            var = 105
            var = 106
            var = 107
            var = 108
            var = 109
            var = 110
            var = 111
            var = 112
            var = 113
            var = 114
            var = 115
            var = 116
            var = 117
            var = 118
            var = 119
            var = 120
            var = 121
            var = 122
            var = 123
            var = 124
            var = 125
            var = 126
            var = 127
            var = 128
            var = 129
            var = 130
            var = 131
            var = 132
            var = 133
            var = 134
            var = 135
            var = 136
            var = 137
            var = 138
            var = 139
            var = 140
            var = 141
            var = 142
            var = 143
            var = 144
            var = 145
            var = 146
            var = 147
            var = 148
            var = 149
            var = 150
            var = 151
            var = 152
            var = 153
            var = 154
            var = 155
            var = 156
            var = 157
            var = 158
            var = 159
            var = 160
            var = 161
            var = 162
            var = 163
            var = 164
            var = 165
            var = 166
            var = 167
            var = 168
            var = 169
            var = 170
            var = 171
            var = 172
            var = 173
            var = 174
            var = 175
            var = 176
            var = 177
            var = 178
            var = 179
            var = 180
            var = 181
            var = 182
            var = 183
            var = 184
            var = 185
            var = 186
            var = 187
            var = 188
            var = 189
            var = 190
            var = 191
            var = 192
            var = 193
            var = 194
            var = 195
            var = 196
            var = 197
            var = 198
            var = 199
            var = 200
            var = 201
            var = 202
            var = 203
            var = 204
            var = 205
            var = 206
            var = 207
            var = 208
            var = 209
            var = 210
            var = 211
            var = 212
            var = 213
            var = 214
            var = 215
            var = 216
            var = 217
            var = 218
            var = 219
            var = 220
            var = 221
            var = 222
            var = 223
            var = 224
            var = 225
            var = 226
            var = 227
            var = 228
            var = 229
            var = 230
            var = 231
            var = 232
            var = 233
            var = 234
            var = 235
            var = 236
            var = 237
            var = 238
            var = 239
            var = 240
            var = 241
            var = 242
            var = 243
            var = 244
            var = 245
            var = 246
            var = 247
            var = 248
            var = 249
            var = 250
            var = 251
            var = 252
            var = 253
            var = 254
            var = 255
            var = 256
            var = 257
            var = 258
            var = 259
            return var
        test.__code__ = ConcreteBytecode.from_code(test.__code__, extended_arg=True).to_code()
        self.assertEqual(test.__code__.co_stacksize, 1)
        self.assertEqual(test(), 259)
    if sys.version_info >= (3, 6):

        def test_fail_extended_arg_jump(self):

            def test():
                var = None
                for _ in range(0, 1):
                    var = 0
                    var = 1
                    var = 2
                    var = 3
                    var = 4
                    var = 5
                    var = 6
                    var = 7
                    var = 8
                    var = 9
                    var = 10
                    var = 11
                    var = 12
                    var = 13
                    var = 14
                    var = 15
                    var = 16
                    var = 17
                    var = 18
                    var = 19
                    var = 20
                    var = 21
                    var = 22
                    var = 23
                    var = 24
                    var = 25
                    var = 26
                    var = 27
                    var = 28
                    var = 29
                    var = 30
                    var = 31
                    var = 32
                    var = 33
                    var = 34
                    var = 35
                    var = 36
                    var = 37
                    var = 38
                    var = 39
                    var = 40
                    var = 41
                    var = 42
                    var = 43
                    var = 44
                    var = 45
                    var = 46
                    var = 47
                    var = 48
                    var = 49
                    var = 50
                    var = 51
                    var = 52
                    var = 53
                    var = 54
                    var = 55
                    var = 56
                    var = 57
                    var = 58
                    var = 59
                    var = 60
                    var = 61
                    var = 62
                    var = 63
                    var = 64
                    var = 65
                    var = 66
                    var = 67
                    var = 68
                    var = 69
                    var = 70
                return var
            bytecode = ConcreteBytecode.from_code(test.__code__, extended_arg=True)
            bytecode.to_code()