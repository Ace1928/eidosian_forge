from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType
from numba.core import errors, utils, serialize
from numba.core.utils import PYVERSION
def _patched_opargs(bc_stream):
    """Patch the bytecode stream.

    - Adds a NOP bytecode at the start to avoid jump target being at the entry.
    """
    yield (0, OPCODE_NOP, None, _FIXED_OFFSET)
    for offset, opcode, arg, nextoffset in bc_stream:
        if opcode in JABS_OPS:
            arg += _FIXED_OFFSET
        yield (offset + _FIXED_OFFSET, opcode, arg, nextoffset + _FIXED_OFFSET)