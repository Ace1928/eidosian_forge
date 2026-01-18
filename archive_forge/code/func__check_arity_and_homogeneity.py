import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def _check_arity_and_homogeneity(sig, args, arity, return_type=None):
    """checks that the following are true:
    - args and sig.args have arg_count elements
    - all input types are homogeneous
    - return type is 'return_type' if provided, otherwise it must be
      homogeneous with the input types.
    """
    assert len(args) == arity
    assert len(sig.args) == arity
    ty = sig.args[0]
    if return_type is None:
        return_type = ty
    if not (all((arg == ty for arg in sig.args)) and sig.return_type == return_type):
        import inspect
        fname = inspect.currentframe().f_back.f_code.co_name
        msg = '{0} called with invalid types: {1}'.format(fname, sig)
        assert False, msg