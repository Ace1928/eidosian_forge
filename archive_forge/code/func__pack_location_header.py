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
@staticmethod
def _pack_location_header(code: int, size: int) -> int:
    return (1 << 7) + (code << 3) + (size - 1 if size <= 8 else 7)