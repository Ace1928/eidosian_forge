import dis
import inspect
import opcode as _opcode
import struct
import sys
import types
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import (
def _assemble_linestable(self, first_lineno, linenos):
    if not linenos:
        return b''
    linetable = []
    old_offset = 0
    iter_in = iter(linenos)
    offset, i_size, old_lineno = next(iter_in)
    old_dlineno = old_lineno - first_lineno
    for offset, i_size, lineno in iter_in:
        dlineno = lineno - old_lineno
        if dlineno == 0:
            continue
        old_lineno = lineno
        doff = offset - old_offset
        old_offset = offset
        self._pack_linetable(doff, old_dlineno, linetable)
        old_dlineno = dlineno
    doff = offset + i_size - old_offset
    self._pack_linetable(doff, old_dlineno, linetable)
    return b''.join(linetable)