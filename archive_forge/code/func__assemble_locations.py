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
def _assemble_locations(self, first_lineno: int, linenos: Iterable[Tuple[int, int, int, Optional[InstrLocation]]]) -> bytes:
    if not linenos:
        return b''
    locations: List[bytearray] = []
    iter_in = iter(linenos)
    _, size, lineno, old_location = next(iter_in)
    old_location = old_location or InstrLocation(lineno, None, None, None)
    lineno = first_lineno
    for _, i_size, new_lineno, location in iter_in:
        location = location or InstrLocation(new_lineno, None, None, None)
        if old_location.lineno and old_location == location:
            size += i_size
            continue
        lineno = self._push_locations(locations, size, lineno, old_location)
        size = i_size
        old_location = location
    self._push_locations(locations, size, lineno, old_location)
    return b''.join(locations)