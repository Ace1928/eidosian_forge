from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType
from numba.core import errors, utils, serialize
from numba.core.utils import PYVERSION
@classmethod
def _compute_lineno(cls, table, code):
    """
        Compute the line numbers for all bytecode instructions.
        """
    for offset, lineno in dis.findlinestarts(code):
        adj_offset = offset + _FIXED_OFFSET
        if adj_offset in table:
            table[adj_offset].lineno = lineno
    known = code.co_firstlineno
    for inst in table.values():
        if inst.lineno >= 0:
            known = inst.lineno
        else:
            inst.lineno = known
    return table