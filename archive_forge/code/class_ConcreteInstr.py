import dis
import inspect
import opcode as _opcode
import struct
import sys
import types
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import (
class ConcreteInstr(Instr):
    """Concrete instruction.

    arg must be an integer in the range 0..2147483647.

    It has a read-only size attribute.
    """
    __slots__ = ('_size', '_extended_args', 'offset')

    def __init__(self, name, arg=UNSET, *, lineno=None, extended_args=None, offset=None):
        self._extended_args = extended_args
        self._set(name, arg, lineno)
        self.offset = offset

    def _check_arg(self, name, opcode, arg):
        if opcode >= _opcode.HAVE_ARGUMENT:
            if arg is UNSET:
                raise ValueError('operation %s requires an argument' % name)
            _check_arg_int(name, arg)
        elif arg is not UNSET:
            raise ValueError('operation %s has no argument' % name)

    def _set(self, name, arg, lineno):
        super()._set(name, arg, lineno)
        size = 2
        if arg is not UNSET:
            while arg > 255:
                size += 2
                arg >>= 8
        if self._extended_args is not None:
            size = 2 + 2 * self._extended_args
        self._size = size

    @property
    def size(self):
        return self._size

    def _cmp_key(self, labels=None):
        return (self._lineno, self._name, self._arg)

    def get_jump_target(self, instr_offset):
        if self._opcode in _opcode.hasjrel:
            s = self._size // 2 if OFFSET_AS_INSTRUCTION else self._size
            return instr_offset + s + self._arg
        if self._opcode in _opcode.hasjabs:
            return self._arg
        return None

    def assemble(self):
        if self._arg is UNSET:
            return bytes((self._opcode, 0))
        arg = self._arg
        b = [self._opcode, arg & 255]
        while arg > 255:
            arg >>= 8
            b[:0] = [_opcode.EXTENDED_ARG, arg & 255]
        if self._extended_args:
            while len(b) < self._size:
                b[:0] = [_opcode.EXTENDED_ARG, 0]
        return bytes(b)

    @classmethod
    def disassemble(cls, lineno, code, offset):
        index = 2 * offset if OFFSET_AS_INSTRUCTION else offset
        op = code[index]
        if op >= _opcode.HAVE_ARGUMENT:
            arg = code[index + 1]
        else:
            arg = UNSET
        name = _opcode.opname[op]
        return cls(name, arg, lineno=lineno, offset=index)