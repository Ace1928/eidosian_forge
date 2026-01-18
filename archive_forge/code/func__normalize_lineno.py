import dis
import inspect
import opcode as _opcode
import struct
import sys
import types
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import (
@staticmethod
def _normalize_lineno(instructions, first_lineno):
    lineno = first_lineno
    for instr in instructions:
        if instr.lineno is not None:
            lineno = instr.lineno
        if isinstance(instr, ConcreteInstr):
            yield (lineno, instr)