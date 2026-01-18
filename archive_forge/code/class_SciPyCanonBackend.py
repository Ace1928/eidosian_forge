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
class SciPyCanonBackend(PythonCanonBackend):

    @staticmethod
    def get_constant_data_from_const(lin_op: LinOp) -> sp.csr_matrix:
        """
        Extract the constant data from a LinOp node of type "*_const".
        """
        constant = sp.csr_matrix(lin_op.data)
        assert constant.shape == lin_op.shape
        return constant

    @staticmethod
    def reshape_constant_data(constant_data: dict[int, sp.csc_matrix], lin_op_shape: tuple[int, int]) -> dict[int, sp.csc_matrix]:
        """
        Reshape constant data from column format to the required shape for operations that
        do not require column format. This function unpacks the constant data dict and reshapes
        the stacked slices of the tensor 'v' according to the lin_op_shape argument.
        """
        return {k: SciPyCanonBackend._reshape_single_constant_tensor(v, lin_op_shape) for k, v in constant_data.items()}

    @staticmethod
    def _reshape_single_constant_tensor(v: sp.csc_matrix, lin_op_shape: tuple[int, int]) -> sp.csc_matrix:
        """
        Given v, which is a matrix of shape (p * lin_op_shape[0] * lin_op_shape[1], 1),
        reshape v into a matrix of shape (p * lin_op_shape[0], lin_op_shape[1]).
        """
        assert v.shape[1] == 1
        p = np.prod(v.shape) // np.prod(lin_op_shape)
        old_shape = (v.shape[0] // p, v.shape[1])
        coo = v.tocoo()
        data, stacked_rows = (coo.data, coo.row)
        slices, rows = np.divmod(stacked_rows, old_shape[0])
        new_cols, new_rows = np.divmod(rows, lin_op_shape[0])
        new_rows = slices * lin_op_shape[0] + new_rows
        new_stacked_shape = (p * lin_op_shape[0], lin_op_shape[1])
        return sp.csc_matrix((data, (new_rows, new_cols)), shape=new_stacked_shape)

    def get_empty_view(self) -> SciPyTensorView:
        """
        Returns an empty view of the corresponding SciPyTensorView subclass,
        coupling the SciPyCanonBackend subclass with the SciPyTensorView subclass.
        """
        return SciPyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col, self.param_to_size, self.param_to_col, self.var_length)

    @staticmethod
    def neg(_lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Given (A, b) in view, return (-A, -b).
        """

        def func(x, _p):
            return -x
        view.apply_all(func)
        return view

    def mul(self, lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Multiply view with constant data from the left.
        When the lhs is parametrized, multiply each slice of the tensor with the 
        single, constant slice of the rhs. 
        Otherwise, multiply the single slice of the tensor with each slice of the rhs.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=False)
        if is_param_free_lhs:
            reps = view.rows // lhs.shape[-1]
            if reps > 1:
                stacked_lhs = sp.kron(sp.eye(reps, format='csr'), lhs)
            else:
                stacked_lhs = lhs

            def func(x, p):
                if p == 1:
                    return (stacked_lhs @ x).tocsr()
                else:
                    return (sp.kron(sp.eye(p, format='csc'), stacked_lhs) @ x).tocsc()
        else:
            reps = view.rows // next(iter(lhs.values())).shape[-1]
            if reps > 1:
                stacked_lhs = self._stacked_kron_r(lhs, reps)
            else:
                stacked_lhs = lhs

            def parametrized_mul(x):
                return {k: v @ x for k, v in stacked_lhs.items()}
            func = parametrized_mul
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

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

    @staticmethod
    def promote(lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Promote view by repeating along axis 0 (rows).
        """
        num_entries = int(np.prod(lin.shape))
        rows = np.zeros(num_entries).astype(int)
        view.select_rows(rows)
        return view

    def mul_elem(self, lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Given (A, b) in view and constant data d, return (A*d, b*d).
        When dealing with parametrized constant data, we need to repeat the variable tensor p times
        and stack them vertically to ensure shape compatibility for elementwise multiplication
        with the parametrized expression.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=True)
        if is_param_free_lhs:

            def func(x, p):
                if p == 1:
                    return lhs.multiply(x)
                else:
                    new_lhs = sp.vstack([lhs] * p)
                    return new_lhs.multiply(x)
        else:

            def parametrized_mul(x):
                return {k: v.multiply(sp.vstack([x] * self.param_to_size[k])) for k, v in lhs.items()}
            func = parametrized_mul
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def sum_entries(_lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Given (A, b) in view, return the sum of the representation
        on the row axis, ie: (sum(A,axis=0), sum(b, axis=0)).
        Here, since the slices are stacked, we sum over the rows corresponding
        to the same slice.
        """

        def func(x, p):
            if p == 1:
                return sp.csr_matrix(x.sum(axis=0))
            else:
                m = x.shape[0] // p
                return (sp.kron(sp.eye(p, format='csc'), np.ones(m)) @ x).tocsc()
        view.apply_all(func)
        return view

    def div(self, lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Given (A, b) in view and constant data d, return (A*(1/d), b*(1/d)).
        d is broadcasted along dimension 1 (columns).
        This function is semantically identical to mul_elem but the view x
        is multiplied with the reciprocal of the lin_op data.

        Note: div currently doesn't support parameters.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=True)
        assert is_param_free_lhs
        lhs.data = np.reciprocal(lhs.data, dtype=float)

        def div_func(x, p):
            if p == 1:
                return lhs.multiply(x)
            else:
                new_lhs = sp.vstack([lhs] * p)
                return new_lhs.multiply(x)
        return view.accumulate_over_variables(div_func, is_param_free_function=is_param_free_lhs)

    @staticmethod
    def diag_vec(lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Diagonal vector to matrix. Given (A, b) with n rows in view, add rows of zeros such that
        the original rows now correspond to the diagonal entries of the n x n expression
        An optional offset parameter `k` can be specified, with k>0 for diagonals above
        the main diagonal, and k<0 for diagonals below the main diagonal.
        """
        assert lin.shape[0] == lin.shape[1]
        k = lin.data
        rows = lin.shape[0]
        total_rows = int(lin.shape[0] ** 2)

        def func(x, p):
            shape = list(x.shape)
            shape[0] = int(total_rows * p)
            x = x.tocoo()
            x_slice, x_row = np.divmod(x.row, x.shape[0] // p)
            if k == 0:
                new_rows = x_row * (rows + 1)
            elif k > 0:
                new_rows = x_row * (rows + 1) + rows * k
            else:
                new_rows = x_row * (rows + 1) - k
            new_rows = (new_rows + x_slice * total_rows).astype(int)
            return sp.csc_matrix((x.data, (new_rows, x.col)), shape)
        view.apply_all(func)
        return view

    @staticmethod
    def get_stack_func(total_rows: int, offset: int) -> Callable:
        """
        Returns a function that takes in a tensor, modifies the shape of the tensor by extending
        it to total_rows, and then shifts the entries by offset along axis 0.
        """

        def stack_func(tensor, p):
            coo_repr = tensor.tocoo()
            m = coo_repr.shape[0] // p
            slices = coo_repr.row // m
            new_rows = coo_repr.row + (slices + 1) * offset
            new_rows = new_rows + slices * (total_rows - m - offset).astype(int)
            return sp.csc_matrix((coo_repr.data, (new_rows, coo_repr.col)), shape=(int(total_rows * p), tensor.shape[1]))
        return stack_func

    def rmul(self, lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
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
            if len(lin.data.shape) == 1 and arg_cols != lhs.shape[0]:
                lhs = lhs.T
            reps = view.rows // lhs.shape[0]
            if reps > 1:
                stacked_lhs = sp.kron(lhs.T, sp.eye(reps, format='csr'))
            else:
                stacked_lhs = lhs.T

            def func(x, p):
                if p == 1:
                    return (stacked_lhs @ x).tocsr()
                else:
                    return (sp.kron(sp.eye(p, format='csc'), stacked_lhs) @ x).tocsc()
        else:
            k, v = next(iter(lhs.items()))
            lhs_rows = v.shape[0] // self.param_to_size[k]
            if len(lin.data.shape) == 1 and arg_cols != lhs_rows:
                lhs = {k: self._transpose_stacked(v, k) for k, v in lhs.items()}
                k, v = next(iter(lhs.items()))
                lhs_rows = v.shape[0] // self.param_to_size[k]
            reps = view.rows // lhs_rows
            lhs = {k: self._transpose_stacked(v, k) for k, v in lhs.items()}
            if reps > 1:
                stacked_lhs = self._stacked_kron_l(lhs, reps)
            else:
                stacked_lhs = lhs

            def parametrized_mul(x):
                return {k: (v @ x).tocsc() for k, v in stacked_lhs.items()}
            func = parametrized_mul
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

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

    @staticmethod
    def trace(lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Select the rows corresponding to the diagonal entries in the expression and sum along
        axis 0.
        Apply kron(eye(p), lhs) to deal with parametrized expressions.
        """
        shape = lin.args[0].shape
        indices = np.arange(shape[0]) * shape[0] + np.arange(shape[0])
        data = np.ones(len(indices))
        idx = (np.zeros(len(indices)), indices.astype(int))
        lhs = sp.csr_matrix((data, idx), shape=(1, np.prod(shape)))

        def func(x, p) -> sp.csc_matrix:
            if p == 1:
                return (lhs @ x).tocsr()
            else:
                return (sp.kron(sp.eye(p, format='csc'), lhs) @ x).tocsc()
        return view.accumulate_over_variables(func, is_param_free_function=True)

    def conv(self, lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Returns view corresponding to a discrete convolution with data 'a', i.e., multiplying from
        the left a repetition of the column vector of 'a' for each column in A, shifted down one row
        after each column, i.e., a Toeplitz matrix.
        If lin_data is a row vector, we must transform the lhs to become a column vector before
        applying the convolution.

        Note: conv currently doesn't support parameters.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=False)
        assert is_param_free_lhs, 'SciPy backend does not support parametrized left operand for conv.'
        assert lhs.ndim == 2
        if len(lin.data.shape) == 1:
            lhs = lhs.T
        rows = lin.shape[0]
        cols = lin.args[0].shape[0]
        nonzeros = lhs.shape[0]
        lhs = lhs.tocoo()
        row_idx = (np.tile(lhs.row, cols) + np.repeat(np.arange(cols), nonzeros)).astype(int)
        col_idx = (np.tile(lhs.col, cols) + np.repeat(np.arange(cols), nonzeros)).astype(int)
        data = np.tile(lhs.data, cols)
        lhs = sp.csr_matrix((data, (row_idx, col_idx)), shape=(rows, cols))

        def func(x, p):
            assert p == 1, 'SciPy backend does not support parametrized right operand for conv.'
            return lhs @ x
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    def kron_r(self, lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Returns view corresponding to Kronecker product of data 'a' with view x, i.e., kron(a,x).
        This function reshapes 'a' into a column vector, computes the Kronecker product with the
        view of x and reorders the row indices afterwards.

        Note: kron_r currently doesn't support parameters.
        """
        lhs, is_param_free_lhs = self.get_constant_data(lin.data, view, column=True)
        assert is_param_free_lhs, 'SciPy backend does not support parametrized left operand for kron_r.'
        assert lhs.ndim == 2
        assert len({arg.shape for arg in lin.args}) == 1
        rhs_shape = lin.args[0].shape
        row_idx = self._get_kron_row_indices(lin.data.shape, rhs_shape)

        def func(x, p):
            assert p == 1, 'SciPy backend does not support parametrized right operand for kron_r.'
            assert x.ndim == 2
            kron_res = sp.kron(lhs, x).tocsr()
            kron_res = kron_res[row_idx, :]
            return kron_res
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_lhs)

    def kron_l(self, lin: LinOp, view: SciPyTensorView) -> SciPyTensorView:
        """
        Returns view corresponding to Kronecker product of view x with data 'a', i.e., kron(x,a).
        This function reshapes 'a' into a column vector, computes the Kronecker product with the
        view of x and reorders the row indices afterwards.

        Note: kron_l currently doesn't support parameters.
        """
        rhs, is_param_free_rhs = self.get_constant_data(lin.data, view, column=True)
        assert is_param_free_rhs, 'SciPy backend does not support parametrized right operand for kron_l.'
        assert rhs.ndim == 2
        assert len({arg.shape for arg in lin.args}) == 1
        lhs_shape = lin.args[0].shape
        row_idx = self._get_kron_row_indices(lhs_shape, lin.data.shape)

        def func(x, p):
            assert p == 1, 'SciPy backend does not support parametrized left operand for kron_l.'
            assert x.ndim == 2
            kron_res = sp.kron(x, rhs).tocsr()
            kron_res = kron_res[row_idx, :]
            return kron_res
        return view.accumulate_over_variables(func, is_param_free_function=is_param_free_rhs)

    def get_variable_tensor(self, shape: tuple[int, ...], variable_id: int) -> dict[int, dict[int, sp.csc_matrix]]:
        """
        Returns tensor of a variable node, i.e., eye(n) across axes 0 and 1, where n is
        the size of the variable.
        This function returns eye(n) in csc format.
        """
        assert variable_id != Constant.ID
        n = int(np.prod(shape))
        return {variable_id: {Constant.ID.value: sp.eye(n, format='csc')}}

    def get_data_tensor(self, data: np.ndarray | sp.spmatrix) -> dict[int, dict[int, sp.csr_matrix]]:
        """
        Returns tensor of constant node as a column vector.
        This function reshapes the data and converts it to csc format.
        """
        if isinstance(data, np.ndarray):
            tensor = sp.csr_matrix(data.reshape((-1, 1), order='F'))
        else:
            tensor = sp.coo_matrix(data).reshape((-1, 1), order='F').tocsr()
        return {Constant.ID.value: {Constant.ID.value: tensor}}

    def get_param_tensor(self, shape: tuple[int, ...], parameter_id: int) -> dict[int, dict[int, sp.csc_matrix]]:
        """
        Returns tensor of a parameter node, i.e., eye(n) across axes 0 and 2, where n is
        the size of the parameter.
        This function returns eye(n).flatten() in csc format.
        """
        assert parameter_id != Constant.ID
        param_size = self.param_to_size[parameter_id]
        shape = (int(np.prod(shape) * param_size), 1)
        arg = (np.ones(param_size), (np.arange(param_size) + np.arange(param_size) * param_size, np.zeros(param_size)))
        param_vec = sp.csc_matrix(arg, shape)
        return {Constant.ID.value: {parameter_id: param_vec}}