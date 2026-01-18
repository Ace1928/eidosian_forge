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
@dataclass
class TensorRepresentation:
    """
    Sparse representation of a 3D Tensor. Semantically similar to COO format, with one extra
    dimension. Here, 'row' is axis 0, 'col' axis 1, and 'parameter_offset' axis 2.
    """
    data: np.ndarray
    row: np.ndarray
    col: np.ndarray
    parameter_offset: np.ndarray
    shape: tuple[int, int]

    def __post_init__(self):
        assert self.data.shape == self.row.shape == self.col.shape == self.parameter_offset.shape

    @classmethod
    def combine(cls, tensors: list[TensorRepresentation]) -> TensorRepresentation:
        """
        Concatenates the row, col, parameter_offset, and data fields of a list of
        TensorRepresentations.
        """
        data, row, col, parameter_offset = (np.array([]), np.array([]), np.array([]), np.array([]))
        for t in tensors:
            data = np.append(data, t.data)
            row = np.append(row, t.row)
            col = np.append(col, t.col)
            parameter_offset = np.append(parameter_offset, t.parameter_offset)
        assert all((t.shape == tensors[0].shape for t in tensors))
        return cls(data, row, col, parameter_offset, tensors[0].shape)

    def __eq__(self, other: TensorRepresentation) -> bool:
        return isinstance(other, TensorRepresentation) and np.all(self.data == other.data) and np.all(self.row == other.row) and np.all(self.col == other.col) and np.all(self.parameter_offset == other.parameter_offset) and (self.shape == other.shape)

    def __add__(self, other: TensorRepresentation) -> TensorRepresentation:
        if self.shape != other.shape:
            raise ValueError('Shapes must match for addition.')
        return TensorRepresentation(np.concatenate([self.data, other.data]), np.concatenate([self.row, other.row]), np.concatenate([self.col, other.col]), np.concatenate([self.parameter_offset, other.parameter_offset]), self.shape)

    @classmethod
    def empty_with_shape(cls, shape: tuple[int, int]) -> TensorRepresentation:
        return cls(np.array([], dtype=float), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), shape)

    def flatten_tensor(self, num_param_slices: int) -> sp.csc_matrix:
        """
        Flatten into 2D scipy csc-matrix in column-major order and transpose.
        """
        rows = self.col.astype(np.int64) * np.int64(self.shape[0]) + self.row.astype(np.int64)
        cols = self.parameter_offset.astype(np.int64)
        shape = (np.int64(np.prod(self.shape)), num_param_slices)
        return sp.csc_matrix((self.data, (rows, cols)), shape=shape)

    def get_param_slice(self, param_offset: int) -> sp.csc_matrix:
        """
        Returns a single slice of the tensor for a given parameter offset.
        """
        mask = self.parameter_offset == param_offset
        return sp.csc_matrix((self.data[mask], (self.row[mask], self.col[mask])), self.shape)