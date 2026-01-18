from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType
from numba.core import errors, utils, serialize
from numba.core.utils import PYVERSION
def _as_opcodes(seq):
    lst = []
    for s in seq:
        c = dis.opmap.get(s)
        if c is not None:
            lst.append(c)
    return lst