import builtins
import operator
import inspect
from functools import cached_property
import llvmlite.ir
from numba.core import types, utils, ir, generators, cgutils
from numba.core.errors import (ForbiddenConstruct, LoweringError,
from numba.core.lowering import BaseLower
def _getvar(self, name, ltype=None):
    if name not in self.varmap:
        self.varmap[name] = self.alloca(name, ltype=ltype)
    return self.varmap[name]