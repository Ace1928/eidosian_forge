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
def _transpose_stacked(self, v: sp.csc_matrix, param_id: int) -> sp.csc_matrix:
    """
        Given v, which is a stacked matrix of shape (p * n, m), transpose each slice of v,
        returning a stacked matrix of shape (p * m, n).
        Example:
        Input:      Output:
        [[A_0],     [[A_0.T],
         [A_1],      [A_1.T],
          ...        ...
        """
    old_shape = (v.shape[0] // self.param_to_size[param_id], v.shape[1])
    p = v.shape[0] // old_shape[0]
    new_shape = (old_shape[1], old_shape[0])
    new_stacked_shape = (p * new_shape[0], new_shape[1])
    v = v.tocoo()
    data, rows, cols = (v.data, v.row, v.col)
    slices, rows = np.divmod(rows, old_shape[0])
    new_rows = cols + slices * new_shape[0]
    new_cols = rows
    return sp.csc_matrix((data, (new_rows, new_cols)), shape=new_stacked_shape)