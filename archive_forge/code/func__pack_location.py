import dis
import inspect
import opcode as _opcode
import struct
import sys
import types
from typing import (
import bytecode as _bytecode
from bytecode.flags import CompilerFlags
from bytecode.instr import (
def _pack_location(self, size: int, lineno: int, location: Optional[InstrLocation]) -> bytearray:
    packed = bytearray()
    l_lineno: Optional[int]
    if location is None:
        l_lineno, end_lineno, col_offset, end_col_offset = (lineno, None, None, None)
    else:
        l_lineno, end_lineno, col_offset, end_col_offset = (location.lineno, location.end_lineno, location.col_offset, location.end_col_offset)
    if l_lineno is None:
        packed.append(self._pack_location_header(15, size))
    elif col_offset is None:
        if end_lineno is not None and end_lineno != l_lineno:
            raise ValueError(f'An instruction cannot have no column offset and span multiple lines (lineno: {l_lineno}, end lineno: {end_lineno}')
        packed.extend((self._pack_location_header(13, size), *self._encode_location_svarint(l_lineno - lineno)))
    else:
        assert end_lineno is not None
        assert end_col_offset is not None
        if end_lineno == l_lineno and l_lineno - lineno == 0 and (col_offset < 72) and (end_col_offset - col_offset <= 15):
            packed.extend((self._pack_location_header(col_offset // 8, size), (col_offset % 8 << 4) + (end_col_offset - col_offset)))
        elif end_lineno == l_lineno and l_lineno - lineno in (1, 2) and (col_offset < 256) and (end_col_offset < 256):
            packed.extend((self._pack_location_header(10 + l_lineno - lineno, size), col_offset, end_col_offset))
        else:
            packed.extend((self._pack_location_header(14, size), *self._encode_location_svarint(l_lineno - lineno), *self._encode_location_varint(end_lineno - l_lineno), *self._encode_location_varint(col_offset + 1), *self._encode_location_varint(end_col_offset + 1)))
    return packed