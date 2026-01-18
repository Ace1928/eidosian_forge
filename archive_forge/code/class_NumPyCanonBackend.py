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
class NumPyCanonBackend(PythonCanonBackend):

    @staticmethod
    def get_constant_data_from_const(lin_op: LinOp) -> np.ndarray:
        """
        Extract the constant data from a LinOp node of type "*_const".
        """
        constant = NumPyCanonBackend._to_dense(lin_op.data)
        assert constant.shape == lin_op.shape
        return constant

    @staticmethod
    def reshape_constant_data(constant_data: dict[int, np.ndarray], lin_op_shape: tuple[int, int]) -> dict[int, np.ndarray]:
        """
        Reshape constant data from column format to the required shape for operations that
        do not require column format. This function unpacks the constant data dict and reshapes
        dimensions 1 and 2 of the tensor 'v' according to the lin_op_shape argument.
        """
        return {k: v.reshape((v.shape[0], *lin_op_shape), order='F') for k, v in constant_data.items()}

    def concatenate_tensors(self, tensors: list[TensorRepresentation]) -> TensorRepresentation:
        """
        Takes list of tensors which have already been offset along axis 0 (rows) and
        combines them into a single tensor.
        """
        return TensorRepresentation.combine(tensors)

    def get_empty_view(self) -> NumPyTensorView:
        """
        Returns an empty view of the corresponding NumPyTensorView subclass,
        coupling the NumPyCanonBackend subclass with the NumPyTensorView subclass.
        """
        return NumPyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col, self.param_to_size, self.param_to_col, self.var_length)

    def mul(self, lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Multiply view with constant data from the left.
        When the lhs is parametrized, multiply each slice of the tensor with the 
        single, constant slice of the rhs. 
        Otherwise, multiply the single slice of the tensor with each slice of the rhs.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=False)
        if isinstance(lhs, dict):
            reps = view.rows // next(iter(lhs.values()))[0].shape[-1]
            stacked_lhs = {k: np.kron(np.eye(reps), v) for k, v in lhs.items()}

            def parametrized_mul(x):
                assert x.shape[0] == 1
                return {k: v @ x for k, v in stacked_lhs.items()}
            func = parametrized_mul
        else:
            assert isinstance(lhs, np.ndarray)
            reps = view.rows // lhs.shape[-1]
            stacked_lhs = np.kron(np.eye(reps), lhs)

            def func(x):
                return stacked_lhs @ x
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def promote(lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Promote view by repeating along axis 1 (rows).
        """
        num_entries = int(np.prod(lin.shape))

        def func(x):
            return np.tile(x, (1, num_entries, 1))
        view.apply_all(func)
        return view

    def mul_elem(self, lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Given (A, b) in view and constant data d, return (A*d, b*d).
        d is broadcasted along dimension 1 (columns).
        When the lhs is parametrized, multiply elementwise each slice of the tensor with the 
        single, constant slice of the rhs. 
        Otherwise, multiply elementwise the single slice of the tensor with each slice of the rhs.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=True)
        if isinstance(lhs, dict):

            def parametrized_mul(x):
                assert x.shape[0] == 1
                return {k: v * x for k, v in lhs.items()}
            func = parametrized_mul
        else:

            def func(x):
                return lhs * x
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def sum_entries(_lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Given (A, b) in view, return the sum of the representation
        on the row axis, ie: (sum(A,axis=1), sum(b, axis=1)).
        """

        def func(x):
            return x.sum(axis=1, keepdims=True)
        view.apply_all(func)
        return view

    def div(self, lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Given (A, b) in view and constant data d, return (A*(1/d), b*(1/d)).
        d is broadcasted along dimension 1 (columns).
        This function is semantically identical to mul_elem but the view x
        is multiplied with the reciprocal of the lin_op data.

        Note: div currently doesn't support parameters.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=True)
        assert is_param_free_lhs
        assert lhs.shape[0] == 1
        lhs = np.reciprocal(lhs, where=lhs != 0, dtype=float)

        def div_func(x):
            return lhs * x
        return view.accumulate_over_variables(div_func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def diag_vec(lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Diagonal vector to matrix. Given (A, b) with n rows in view, add rows of zeros such that
        the original rows now correspond to the diagonal entries of the n x n expression.
        An optional offset parameter `k` can be specified, with k>0 for diagonals above
        the main diagonal, and k<0 for diagonals below the main diagonal.
        """
        assert lin.shape[0] == lin.shape[1]
        k = lin.data
        rows = lin.shape[0]
        total_rows = int(lin.shape[0] ** 2)

        def func(x):
            x_rows = x.shape[1]
            shape = list(x.shape)
            shape[1] = total_rows
            if k == 0:
                new_rows = np.arange(x_rows) * (rows + 1)
            elif k > 0:
                new_rows = np.arange(x_rows) * (rows + 1) + rows * k
            else:
                new_rows = np.arange(x_rows) * (rows + 1) - k
            matrix = np.zeros(shape)
            matrix[:, new_rows, :] = x
            return matrix
        view.apply_all(func)
        return view

    @staticmethod
    def get_stack_func(total_rows: int, offset: int) -> Callable:
        """
        Returns a function that takes in a tensor, modifies the shape of the tensor by extending
        it to total_rows, and then shifts the entries by offset along axis 1.
        """

        def stack_func(tensor):
            rows = tensor.shape[1]
            new_rows = (np.arange(rows) + offset).astype(int)
            matrix = np.zeros(shape=(tensor.shape[0], int(total_rows), tensor.shape[2]))
            matrix[:, new_rows, :] = tensor
            return matrix
        return stack_func

    def rmul(self, lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Multiply view with constant data from the right.
        When the rhs is parametrized, multiply each slice of the tensor with the
        single, constant slice of the lhs.
        Otherwise, multiply the single slice of the tensor with each slice of the lhs.

        Note: Even though this is rmul, we still use "lhs", as is implemented via a
        multiplication from the left in this function.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=False)
        arg_cols = lin.args[0].shape[0] if len(lin.args[0].shape) == 1 else lin.args[0].shape[1]
        if is_param_free_lhs:
            lhs_rows = lhs.shape[-2]
            if len(lin.data.shape) == 1 and arg_cols != lhs_rows:
                lhs = np.swapaxes(lhs, -2, -1)
            lhs_rows = lhs.shape[-2]
            reps = view.rows // lhs_rows
            lhs_transposed = np.swapaxes(lhs, -2, -1)
            stacked_lhs = np.kron(lhs_transposed, np.eye(reps))

            def func(x):
                return stacked_lhs @ x
        else:
            lhs_shape = next(iter(lhs.values()))[0].shape
            lhs_rows = lhs_shape[-2]
            if len(lin.data.shape) == 1 and arg_cols != lhs_rows:
                lhs = {k: np.swapaxes(v, -2, -1) for k, v in lhs.items()}
                lhs_shape = next(iter(lhs.values()))[0].shape
            lhs_rows = lhs_shape[-2]
            reps = view.rows // lhs_rows
            stacked_lhs = {k: np.kron(np.swapaxes(v, -2, -1), np.eye(reps)) for k, v in lhs.items()}

            def parametrized_mul(x):
                assert x.shape[0] == 1
                return {k: v @ x for k, v in stacked_lhs.items()}
            func = parametrized_mul
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def trace(lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Select the rows corresponding to the diagonal entries in the expression and sum along
        axis 0.
        """
        shape = lin.args[0].shape
        indices = np.arange(shape[0]) * shape[0] + np.arange(shape[0])
        lhs = np.zeros(shape=(1, np.prod(shape)))
        lhs[0, indices] = 1

        def func(x):
            return lhs @ x
        return view.accumulate_over_variables(func, is_param_free_function=True)

    def conv(self, lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Returns view corresponding to a discrete convolution with data 'a', i.e., multiplying from
        the left a repetition of the column vector of 'a' for each column in A, shifted down one row
        after each column, i.e., a Toeplitz matrix.
        If lin_data is a row vector, we must transform the lhs to become a column vector before
        applying the convolution.

        Note: conv currently doesn't support parameters.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=False)
        assert is_param_free_lhs
        if len(lin.data.shape) == 1:
            lhs = np.swapaxes(lhs, -2, -1)

        def func(x):
            return convolve(lhs, x)
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    def kron_r(self, lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Returns view corresponding to Kronecker product of data 'a' with view x, i.e., kron(a,x).
        This function reshapes 'a' into a column vector, computes the Kronecker product with the
        view of x and reorders the row indices afterwards.

        Note: kron_r currently doesn't support parameters.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=True)
        assert is_param_free_lhs
        assert len(lhs) == 1
        lhs = lhs[0]
        assert lhs.ndim == 2
        assert len({arg.shape for arg in lin.args}) == 1
        rhs_shape = lin.args[0].shape
        row_idx = self._get_kron_row_indices(lin.data.shape, rhs_shape)

        def func(x: np.ndarray) -> np.ndarray:
            assert x.ndim == 3
            kron_res = np.kron(lhs, x)
            kron_res = kron_res[:, row_idx, :]
            return kron_res
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    def kron_l(self, lin: LinOp, view: NumPyTensorView) -> NumPyTensorView:
        """
        Returns view corresponding to Kronecker product of view x with data 'a', i.e., kron(x,a).
        This function reshapes 'a' into a column vector, computes the Kronecker product with the
        view of x and reorders the row indices afterwards.

        Note: kron_l currently doesn't support parameters.
        """
        rhs, is_param_free_rhs = self.get_constant_data(lin.data, view, column=True)
        assert is_param_free_rhs
        assert len(rhs) == 1
        rhs = rhs[0]
        assert rhs.ndim == 2
        assert len({arg.shape for arg in lin.args}) == 1
        lhs_shape = lin.args[0].shape
        row_idx = self._get_kron_row_indices(lhs_shape, lin.data.shape)

        def func(x: np.ndarray) -> np.ndarray:
            assert x.ndim == 3
            kron_res = np.kron(x, rhs)
            kron_res = kron_res[:, row_idx, :]
            return kron_res
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_rhs)

    def get_variable_tensor(self, shape: tuple[int, ...], variable_id: int) -> dict[int, dict[int, np.ndarray]]:
        """
        Returns tensor of a variable node, i.e., eye(n) across axes 0 and 1, where n is
        the size of the variable.
        This function expands the dimension of an identity matrix of size n on the parameter axis.
        """
        assert variable_id != Constant.ID
        n = int(np.prod(shape))
        return {variable_id: {Constant.ID.value: np.expand_dims(np.eye(n), axis=0)}}

    def get_data_tensor(self, data: np.ndarray) -> dict[int, dict[int, np.ndarray]]:
        """
        Returns tensor of constant node as a column vector.
        This function expands the dimension of the column vector on the parameter axis.
        """
        data = self._to_dense(data)
        tensor = data.reshape((-1, 1), order='F')
        return {Constant.ID.value: {Constant.ID.value: np.expand_dims(tensor, axis=0)}}

    def get_param_tensor(self, shape: tuple[int, ...], parameter_id: int) -> dict[int, dict[int, np.ndarray]]:
        """
        Returns tensor of a parameter node, i.e., eye(n) across axes 0 and 2, where n is
        the size of the parameter.
        This function expands the dimension of an identity matrix of size n on the column axis.
        """
        assert parameter_id != Constant.ID
        n = int(np.prod(shape))
        return {Constant.ID.value: {parameter_id: np.expand_dims(np.eye(n), axis=-1)}}

    @staticmethod
    def _to_dense(x):
        """
        Internal function that converts a sparse input to a dense numpy array.
        """
        try:
            res = x.toarray()
        except AttributeError:
            res = x
        res = np.atleast_2d(res)
        return res