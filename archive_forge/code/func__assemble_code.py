import dis
import inspect
import opcode as _opcode
import struct
import sys
import types
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import (
def _assemble_code(self):
    offset = 0
    code_str = []
    linenos = []
    for lineno, instr in self._normalize_lineno(self, self.first_lineno):
        code_str.append(instr.assemble())
        i_size = instr.size
        linenos.append((offset * 2 if OFFSET_AS_INSTRUCTION else offset, i_size, lineno))
        offset += i_size // 2 if OFFSET_AS_INSTRUCTION else i_size
    code_str = b''.join(code_str)
    return (code_str, linenos)