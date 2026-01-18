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
class SciPyTensorView(DictTensorView):

    @property
    def rows(self) -> int:
        """
        Number of rows of the TensorView.
        This is calculated by dividing the totals rows of the tensor by the
        number of parameter slices.
        """
        if self.tensor is not None:
            for param_dict in self.tensor.values():
                for param_id, param_mat in param_dict.items():
                    return param_mat.shape[0] // self.param_to_size[param_id]
        else:
            raise ValueError('Tensor cannot be None')

    def get_tensor_representation(self, row_offset: int, total_rows: int) -> TensorRepresentation:
        """
        Returns a TensorRepresentation of [A b] tensor.
        This function iterates through all the tensor data and constructs their
        respective representation in COO format. The row data is adjusted according
        to the position of each element within a parameter slice. The parameter_offset
        finds which slice the original row indices belong to before applying the column
        offset.
        """
        assert self.tensor is not None
        shape = (total_rows, self.var_length + 1)
        tensor_representations = []
        for variable_id, variable_tensor in self.tensor.items():
            for parameter_id, parameter_matrix in variable_tensor.items():
                p = self.param_to_size[parameter_id]
                m = parameter_matrix.shape[0] // p
                coo_repr = parameter_matrix.tocoo(copy=False)
                tensor_representations.append(TensorRepresentation(coo_repr.data, coo_repr.row % m + row_offset, coo_repr.col + self.id_to_col[variable_id], coo_repr.row // m + np.ones(coo_repr.nnz) * self.param_to_col[parameter_id], shape=shape))
        return TensorRepresentation.combine(tensor_representations)

    def select_rows(self, rows: np.ndarray) -> None:
        """
        Select 'rows' from tensor. If there are multiple parameters 'p',
        we must select the same 'rows' from each parameter slice. This is done by
        introducing an offset of size 'm' for every parameter.
        """

        def func(x, p):
            if p == 1:
                return x[rows, :]
            else:
                m = x.shape[0] // p
                return x[np.tile(rows, p) + np.repeat(np.arange(p) * m, len(rows)), :]
        self.apply_all(func)

    def apply_all(self, func: Callable) -> None:
        """
        Apply 'func' across all variables and parameter slices.
        For the stacked-slices backend, we must pass an additional parameter 'p'
        which is the number of parameter slices.
        """
        self.tensor = {var_id: {k: func(v, self.param_to_size[k]) for k, v in parameter_repr.items()} for var_id, parameter_repr in self.tensor.items()}

    def create_new_tensor_view(self, variable_ids: set[int], tensor: Any, is_parameter_free: bool) -> SciPyTensorView:
        """
        Create new SciPyTensorView with same shape information as self,
        but new tensor data.
        """
        return SciPyTensorView(variable_ids, tensor, is_parameter_free, self.param_size_plus_one, self.id_to_col, self.param_to_size, self.param_to_col, self.var_length)

    def apply_to_parameters(self, func: Callable, parameter_representation: dict[int, sp.spmatrix]) -> dict[int, sp.spmatrix]:
        """
        Apply 'func' to each slice of the parameter representation.
        For the stacked-slices backend, we must pass an additional parameter 'p'
        which is the number of parameter slices.
        """
        return {k: func(v, self.param_to_size[k]) for k, v in parameter_representation.items()}

    @staticmethod
    def add_tensors(a: sp.spmatrix, b: sp.spmatrix) -> sp.spmatrix:
        """
        Apply element-wise summation on two sparse matrices.
        """
        return a + b

    @staticmethod
    def tensor_type():
        """
        The tensor representation of the stacked slices backend is one big
        sparse matrix instead of smaller sparse matrices in a list.
        """
        return sp.spmatrix