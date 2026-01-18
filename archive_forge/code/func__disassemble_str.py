import sys
import types
import collections
import io
from opcode import *
from opcode import (
def _disassemble_str(source, **kwargs):
    """Compile the source string, then disassemble the code object."""
    _disassemble_recursive(_try_compile(source, '<dis>'), **kwargs)