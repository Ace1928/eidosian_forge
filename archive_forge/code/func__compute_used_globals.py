from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType
from numba.core import errors, utils, serialize
from numba.core.utils import PYVERSION
@classmethod
def _compute_used_globals(cls, func, table, co_consts, co_names):
    """
        Compute the globals used by the function with the given
        bytecode table.
        """
    d = {}
    globs = func.__globals__
    builtins = globs.get('__builtins__', utils.builtins)
    if isinstance(builtins, ModuleType):
        builtins = builtins.__dict__
    for inst in table.values():
        if inst.opname == 'LOAD_GLOBAL':
            name = co_names[_fix_LOAD_GLOBAL_arg(inst.arg)]
            if name not in d:
                try:
                    value = globs[name]
                except KeyError:
                    value = builtins[name]
                d[name] = value
    for co in co_consts:
        if isinstance(co, CodeType):
            subtable = OrderedDict(ByteCodeIter(co))
            d.update(cls._compute_used_globals(func, subtable, co.co_consts, co.co_names))
    return d