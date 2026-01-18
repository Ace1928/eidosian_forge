import sys
import types
import collections
import io
from opcode import *
from opcode import (
def _disassemble_bytes(code, lasti=-1, varname_from_oparg=None, names=None, co_consts=None, linestarts=None, *, file=None, line_offset=0, exception_entries=(), co_positions=None, show_caches=False):
    show_lineno = bool(linestarts)
    if show_lineno:
        maxlineno = max(linestarts.values()) + line_offset
        if maxlineno >= 1000:
            lineno_width = len(str(maxlineno))
        else:
            lineno_width = 3
    else:
        lineno_width = 0
    maxoffset = len(code) - 2
    if maxoffset >= 10000:
        offset_width = len(str(maxoffset))
    else:
        offset_width = 4
    for instr in _get_instructions_bytes(code, varname_from_oparg, names, co_consts, linestarts, line_offset=line_offset, exception_entries=exception_entries, co_positions=co_positions, show_caches=show_caches):
        new_source_line = show_lineno and instr.starts_line is not None and (instr.offset > 0)
        if new_source_line:
            print(file=file)
        is_current_instr = instr.offset == lasti
        print(instr._disassemble(lineno_width, is_current_instr, offset_width), file=file)
    if exception_entries:
        print('ExceptionTable:', file=file)
        for entry in exception_entries:
            lasti = ' lasti' if entry.lasti else ''
            end = entry.end - 2
            print(f'  {entry.start} to {end} -> {entry.target} [{entry.depth}]{lasti}', file=file)