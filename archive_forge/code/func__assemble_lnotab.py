import dis
import inspect
import opcode as _opcode
import struct
import sys
import types
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import (
@staticmethod
def _assemble_lnotab(first_lineno, linenos):
    lnotab = []
    old_offset = 0
    old_lineno = first_lineno
    for offset, _, lineno in linenos:
        dlineno = lineno - old_lineno
        if dlineno == 0:
            continue
        if dlineno < 0 and sys.version_info < (3, 6):
            raise ValueError('negative line number delta is not supported on Python < 3.6')
        old_lineno = lineno
        doff = offset - old_offset
        old_offset = offset
        while doff > 255:
            lnotab.append(b'\xff\x00')
            doff -= 255
        while dlineno < -128:
            lnotab.append(struct.pack('Bb', doff, -128))
            doff = 0
            dlineno -= -128
        while dlineno > 127:
            lnotab.append(struct.pack('Bb', doff, 127))
            doff = 0
            dlineno -= 127
        assert 0 <= doff <= 255
        assert -128 <= dlineno <= 127
        lnotab.append(struct.pack('Bb', doff, dlineno))
    return b''.join(lnotab)