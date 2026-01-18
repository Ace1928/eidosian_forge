import math
import operator
import sys
import numpy as np
import llvmlite.ir
from llvmlite.ir import Constant
from numba.core.imputils import Registry, impl_ret_untracked
from numba import typeof
from numba.core import types, utils, config, cgutils
from numba.core.extending import overload
from numba.core.typing import signature
from numba.cpython.unsafe.numbers import trailing_zeros
@overload(_unsigned)
def _unsigned_impl(T):
    if T in types.unsigned_domain:
        return lambda T: T
    elif T in types.signed_domain:
        newT = getattr(types, 'uint{}'.format(T.bitwidth))
        return lambda T: newT(T)