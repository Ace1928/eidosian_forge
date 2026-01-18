from collections import namedtuple, OrderedDict
import dis
import inspect
import itertools
from types import CodeType, ModuleType
from numba.core import errors, utils, serialize
from numba.core.utils import PYVERSION
def _fix_LOAD_GLOBAL_arg(arg):
    if PYVERSION in ((3, 11), (3, 12)):
        return arg >> 1
    elif PYVERSION in ((3, 9), (3, 10)):
        return arg
    else:
        raise NotImplementedError(PYVERSION)