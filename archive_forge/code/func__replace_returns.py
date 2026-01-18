import types as pytypes  # avoid confusion with numba.types
import copy
import ctypes
import numba.core.analysis
from numba.core import types, typing, errors, ir, rewrites, config, ir_utils
from numba.parfors.parfor import internal_prange
from numba.core.ir_utils import (
from numba.core.analysis import (
from numba.core import postproc
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty_inferred
import numpy as np
import operator
import numba.misc.special
def _replace_returns(blocks, target, return_label):
    """
    Return return statement by assigning directly to target, and a jump.
    """
    for label, block in blocks.items():
        casts = []
        for i in range(len(block.body)):
            stmt = block.body[i]
            if isinstance(stmt, ir.Return):
                assert i + 1 == len(block.body)
                block.body[i] = ir.Assign(stmt.value, target, stmt.loc)
                block.body.append(ir.Jump(return_label, stmt.loc))
                for cast in casts:
                    if cast.target.name == stmt.value.name:
                        cast.value = cast.value.value
            elif isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and (stmt.value.op == 'cast'):
                casts.append(stmt)