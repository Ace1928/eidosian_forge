from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable
import numpy as np
import scipy.sparse as sp
from scipy.signal import convolve
from cvxpy.lin_ops import LinOp
from cvxpy.settings import (
def _stacked_kron_r(self, lhs: dict[int, list[sp.csc_matrix]], reps: int) -> sp.csc_matrix:
    """
        Given a stacked lhs
        [[A_0],
         [A_1],
         ...
        apply the Kronecker product with the identity matrix of size reps
        (kron(eye(reps), lhs)) to each slice, e.g., for reps = 2:
        [[A_0, 0],
         [0, A_0],
         [A_1, 0],
         [0, A_1],
         ...
        """
    res = dict()
    for param_id, v in lhs.items():
        p = self.param_to_size[param_id]
        old_shape = (v.shape[0] // p, v.shape[1])
        coo = v.tocoo()
        data, rows, cols = (coo.data, coo.row, coo.col)
        slices, rows = np.divmod(rows, old_shape[0])
        new_rows = np.repeat(rows + slices * old_shape[0] * reps, reps) + np.tile(np.arange(reps) * old_shape[0], len(rows))
        new_cols = np.repeat(cols, reps) + np.tile(np.arange(reps) * old_shape[1], len(cols))
        new_data = np.repeat(data, reps)
        new_shape = (v.shape[0] * reps, v.shape[1] * reps)
        res[param_id] = sp.csc_matrix((new_data, (new_rows, new_cols)), shape=new_shape)
    return res