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
@staticmethod
def _get_kron_row_indices(lhs_shape, rhs_shape):
    """
        Internal function that computes the row indices corresponding to the
        kronecker product of two sparse tensors.
        """
    rhs_ones = np.ones(rhs_shape)
    lhs_ones = np.ones(lhs_shape)
    rhs_arange = np.arange(np.prod(rhs_shape)).reshape(rhs_shape, order='F')
    lhs_arange = np.arange(np.prod(lhs_shape)).reshape(lhs_shape, order='F')
    row_indices = (np.kron(lhs_ones, rhs_arange) + np.kron(lhs_arange, rhs_ones * np.prod(rhs_shape))).flatten(order='F').astype(int)
    return row_indices