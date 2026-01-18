import dis
import inspect
import opcode as _opcode
import struct
import sys
import types
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import (
@staticmethod
def _remove_extended_args(instructions):
    nb_extended_args = 0
    extended_arg = None
    index = 0
    while index < len(instructions):
        instr = instructions[index]
        if isinstance(instr, SetLineno):
            index += 1
            continue
        if instr.name == 'EXTENDED_ARG':
            nb_extended_args += 1
            if extended_arg is not None:
                extended_arg = (extended_arg << 8) + instr.arg
            else:
                extended_arg = instr.arg
            del instructions[index]
            continue
        if extended_arg is not None:
            arg = (extended_arg << 8) + instr.arg
            extended_arg = None
            instr = ConcreteInstr(instr.name, arg, lineno=instr.lineno, extended_args=nb_extended_args, offset=instr.offset)
            instructions[index] = instr
            nb_extended_args = 0
        index += 1
    if extended_arg is not None:
        raise ValueError('EXTENDED_ARG at the end of the code')