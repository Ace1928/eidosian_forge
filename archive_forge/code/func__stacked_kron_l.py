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
def _stacked_kron_l(self, lhs: dict[int, list[sp.csc_matrix]], reps: int) -> sp.csc_matrix:
    """
        Given a stacked lhs with the following entries:
        [[a11, a12],
         [a21, a22],
         ...
        Apply the Kronecker product with the identity matrix of size reps
        (kron(lhs, eye(reps))) to each slice, e.g., for reps = 2:
        [[a11, 0, a12, 0],
         [0, a11, 0, a12],
         [a21, 0, a22, 0],
         [0, a21, 0, a22],
         ...
        """
    res = dict()
    for param_id, v in lhs.items():
        self.param_to_size[param_id]
        coo = v.tocoo()
        data, rows, cols = (coo.data, coo.row, coo.col)
        new_rows = np.repeat(rows * reps, reps) + np.tile(np.arange(reps), len(rows))
        new_cols = np.repeat(cols * reps, reps) + np.tile(np.arange(reps), len(cols))
        new_data = np.repeat(data, reps)
        new_shape = (v.shape[0] * reps, v.shape[1] * reps)
        res[param_id] = sp.csc_matrix((new_data, (new_rows, new_cols)), shape=new_shape)
    return res