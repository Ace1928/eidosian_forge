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
class PythonCanonBackend(CanonBackend):
    """
    Each tensor has 3 dimensions. The first one is the parameter axis, the second one is the rows
    and the third one is the variable columns.

    For example:
    - A new variable of size n has shape (1, n, n)
    - A new parameter of size n has shape (n, n, 1)
    - A new constant of size n has shape (1, n, 1)
    """

    def build_matrix(self, lin_ops: list[LinOp]) -> sp.csc_matrix:
        self.id_to_col[-1] = self.var_length
        constraint_res = []
        total_rows = sum((np.prod(lin_op.shape) for lin_op in lin_ops))
        row_offset = 0
        for lin_op in lin_ops:
            lin_op_rows = np.prod(lin_op.shape)
            empty_view = self.get_empty_view()
            lin_op_tensor = self.process_constraint(lin_op, empty_view)
            constraint_res.append(lin_op_tensor.get_tensor_representation(row_offset, total_rows))
            row_offset += lin_op_rows
        tensor_res = self.concatenate_tensors(constraint_res)
        self.id_to_col.pop(-1)
        return tensor_res.flatten_tensor(self.param_size_plus_one)

    def process_constraint(self, lin_op: LinOp, empty_view: TensorView) -> TensorView:
        """
        Depth-first parsing of a linOp node.

        Parameters
        ----------
        lin_op: a node in the linOp tree.
        empty_view: TensorView used to create tensors for leaf nodes.

        Returns
        -------
        The processed node as a TensorView.
        """
        if lin_op.type == 'variable':
            assert isinstance(lin_op.data, int)
            assert len(lin_op.shape) in {0, 1, 2}
            variable_tensor = self.get_variable_tensor(lin_op.shape, lin_op.data)
            return empty_view.create_new_tensor_view({lin_op.data}, variable_tensor, is_parameter_free=True)
        elif lin_op.type in {'scalar_const', 'dense_const', 'sparse_const'}:
            data_tensor = self.get_data_tensor(lin_op.data)
            return empty_view.create_new_tensor_view({Constant.ID.value}, data_tensor, is_parameter_free=True)
        elif lin_op.type == 'param':
            param_tensor = self.get_param_tensor(lin_op.shape, lin_op.data)
            return empty_view.create_new_tensor_view({Constant.ID.value}, param_tensor, is_parameter_free=False)
        else:
            func = self.get_func(lin_op.type)
            if lin_op.type in {'vstack', 'hstack'}:
                return func(lin_op, empty_view)
            res = None
            for arg in lin_op.args:
                arg_coeff = self.process_constraint(arg, empty_view)
                arg_res = func(lin_op, arg_coeff)
                if res is None:
                    res = arg_res
                else:
                    res += arg_res
            assert res is not None
            return res

    def get_constant_data(self, lin_op: LinOp, view: TensorView, column: bool) -> tuple[np.ndarray | sp.spmatrix, bool]:
        """
        Extract the constant data from a LinOp node. In most cases, lin_op will be of
        type "*_const" or "param", but can handle arbitrary types.
        """
        constants = {'scalar_const', 'dense_const', 'sparse_const'}
        if not column and lin_op.type in constants and (len(lin_op.shape) == 2):
            constant_data = self.get_constant_data_from_const(lin_op)
            return (constant_data, True)
        constant_view = self.process_constraint(lin_op, view)
        assert constant_view.variable_ids == {Constant.ID.value}
        constant_data = constant_view.tensor[Constant.ID.value]
        if not column and len(lin_op.shape) >= 1:
            lin_op_shape = lin_op.shape if len(lin_op.shape) == 2 else [1, lin_op.shape[0]]
            constant_data = self.reshape_constant_data(constant_data, lin_op_shape)
        data_to_return = constant_data[Constant.ID.value] if constant_view.is_parameter_free else constant_data
        return (data_to_return, constant_view.is_parameter_free)

    @staticmethod
    @abstractmethod
    def get_constant_data_from_const(lin_op: LinOp) -> np.ndarray | sp.spmatrix:
        """
        Extract the constant data from a LinOp node of type "*_const".
        """
        pass

    @staticmethod
    @abstractmethod
    def reshape_constant_data(constant_data: Any, lin_op_shape: tuple[int, int]) -> Any:
        """
        Reshape constant data from column format to the required shape for operations that
        do not require column format
        """
        pass

    @staticmethod
    def concatenate_tensors(tensors: list[TensorRepresentation]) -> TensorRepresentation:
        """
        Takes list of tensors which have already been offset along axis 0 (rows) and
        combines them into a single tensor.
        """
        return TensorRepresentation.combine(tensors)

    @abstractmethod
    def get_empty_view(self) -> TensorView:
        """
        Returns an empty view of the corresponding TensorView subclass, coupling the CanonBackend
        subclass with the TensorView subclass.
        """
        pass

    def get_func(self, func_name: str) -> Callable:
        """
        Map the name of a function as given by the linOp to the implementation.

        Parameters
        ----------
        func_name: The name of the function.

        Returns
        -------
        The function implementation.
        """
        mapping = {'sum': self.sum_op, 'mul': self.mul, 'promote': self.promote, 'neg': self.neg, 'mul_elem': self.mul_elem, 'sum_entries': self.sum_entries, 'div': self.div, 'reshape': self.reshape, 'index': self.index, 'diag_vec': self.diag_vec, 'hstack': self.hstack, 'vstack': self.vstack, 'transpose': self.transpose, 'upper_tri': self.upper_tri, 'diag_mat': self.diag_mat, 'rmul': self.rmul, 'trace': self.trace, 'conv': self.conv, 'kron_l': self.kron_l, 'kron_r': self.kron_r}
        return mapping[func_name]

    @staticmethod
    def sum_op(_lin: LinOp, view: TensorView) -> TensorView:
        """
        Sum (along axis 1) is implicit in Ax+b, so it is a NOOP.
        """
        return view

    @staticmethod
    def reshape(_lin: LinOp, view: TensorView) -> TensorView:
        """
        Reshaping only changes the shape attribute of the LinOp, so it is a NOOP.
        """
        return view

    @abstractmethod
    def mul(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Multiply view with constant data from the left.
        When the lhs is parametrized, multiply each slice of the tensor with the 
        single, constant slice of the rhs. 
        Otherwise, multiply the single slice of the tensor with each slice of the rhs.
        """
        pass

    @staticmethod
    @abstractmethod
    def promote(lin: LinOp, view: TensorView) -> TensorView:
        """
        Promote view by repeating along axis 0 (rows)
        """
        pass

    @staticmethod
    def neg(_lin: LinOp, view: TensorView) -> TensorView:
        """
        Given (A, b) in view, return (-A, -b).
        """

        def func(x):
            return -x
        view.apply_all(func)
        return view

    @abstractmethod
    def mul_elem(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Given (A, b) in view and constant data d, return (A*d, b*d).
        d is broadcasted along dimension 1 (columns).
        When the lhs is parametrized, multiply elementwise each slice of the tensor with the 
        single, constant slice of the rhs. 
        Otherwise, multiply elementwise the single slice of the tensor with each slice of the rhs.
        """
        pass

    @staticmethod
    @abstractmethod
    def sum_entries(_lin: LinOp, view: TensorView) -> TensorView:
        """
        Given (A, b) in view, return (sum(A,axis=0), sum(b, axis=0))
        """
        pass

    @abstractmethod
    def div(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Given (A, b) in view and constant data d, return (A*(1/d), b*(1/d)).
        d is broadcasted along dimension 1 (columns)
        This function is semantically identical to mul_elem but the view x
        is multiplied with the reciprocal of the lin_op data instead.

        Note: div currently doesn't support parameters.
        """
        pass

    @staticmethod
    def index(lin: LinOp, view: TensorView) -> TensorView:
        """
        Given (A, b) in view, select the rows corresponding to the elements of the expression being
        indexed.
        """
        indices = [np.arange(s.start, s.stop, s.step) for s in lin.data]
        if len(indices) == 1:
            rows = indices[0]
        elif len(indices) == 2:
            rows = np.add.outer(indices[0], indices[1] * lin.args[0].shape[0]).flatten(order='F')
        else:
            raise ValueError
        view.select_rows(rows)
        return view

    @staticmethod
    @abstractmethod
    def diag_vec(lin: LinOp, view: TensorView) -> TensorView:
        """
        Diagonal vector to matrix. Given (A, b) with n rows in view, add rows of zeros such that
        the original rows now correspond to the diagonal entries of the n x n expression
        An optional offset parameter `k` can be specified, with k>0 for diagonals above
        the main diagonal, and k<0 for diagonals below the main diagonal.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_stack_func(total_rows: int, offset: int) -> Callable:
        """
        Returns a function that takes in a tensor, modifies the shape of the tensor by extending
        it to total_rows, and then shifts the entries by offset along axis 0.
        """
        pass

    def hstack(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Given views (A0,b0), (A1,b1),..., (An,bn), stack all tensors along axis 0,
        i.e., return
        (A0, b0)
         A1, b1
         ...
         An, bn.
        """
        offset = 0
        total_rows = sum((np.prod(arg.shape) for arg in lin.args))
        res = None
        for arg in lin.args:
            arg_view = self.process_constraint(arg, view)
            func = self.get_stack_func(total_rows, offset)
            arg_view.apply_all(func)
            offset += np.prod(arg.shape)
            if res is None:
                res = arg_view
            else:
                res += arg_view
        assert res is not None
        return res

    def vstack(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Given views (A0,b0), (A1,b1),..., (An,bn), first, stack them along axis 0 via hstack.
        Then, permute the rows of the resulting tensor to be consistent with stacking the arguments
        vertically instead of horizontally.
        """
        view = self.hstack(lin, view)
        offset = 0
        indices = []
        for arg in lin.args:
            arg_rows = np.prod(arg.shape)
            indices.append(np.arange(arg_rows).reshape(arg.shape, order='F') + offset)
            offset += arg_rows
        order = np.vstack(indices).flatten(order='F').astype(int)
        view.select_rows(order)
        return view

    @staticmethod
    def transpose(lin: LinOp, view: TensorView) -> TensorView:
        """
        Given (A, b) in view, permute the rows such that they correspond to the transposed
        expression.
        """
        rows = np.arange(np.prod(lin.shape)).reshape(lin.shape).flatten(order='F')
        view.select_rows(rows)
        return view

    @staticmethod
    def upper_tri(lin: LinOp, view: TensorView) -> TensorView:
        """
        Given (A, b) in view, select the rows corresponding to the elements above the diagonal
        in the original expression.
        Note: The diagonal itself is not included.
        """
        indices = np.arange(np.prod(lin.args[0].shape)).reshape(lin.args[0].shape, order='F')
        triu_indices = indices[np.triu_indices_from(indices, k=1)]
        view.select_rows(triu_indices)
        return view

    @staticmethod
    def diag_mat(lin: LinOp, view: TensorView) -> TensorView:
        """
        Diagonal matrix to vector. Given (A, b) in view, select the rows corresponding to the
        elements on the diagonal in the original expression.
        An optional offset parameter `k` can be specified, with k>0 for diagonals above
        the main diagonal, and k<0 for diagonals below the main diagonal.
        """
        rows = lin.shape[0]
        k = lin.data
        original_rows = rows + abs(k)
        if k == 0:
            diag_indices = np.arange(rows) * (rows + 1)
        elif k > 0:
            diag_indices = np.arange(rows) * (original_rows + 1) + original_rows * k
        else:
            diag_indices = np.arange(rows) * (original_rows + 1) - k
        view.select_rows(diag_indices.astype(int))
        return view

    @abstractmethod
    def rmul(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Multiply view with constant data from the right.
        When the rhs is parametrized, multiply each slice of the tensor with the
        single, constant slice of the lhs.
        Otherwise, multiply the single slice of the tensor with each slice of the lhs.
        """
        pass

    @staticmethod
    @abstractmethod
    def trace(lin: LinOp, view: TensorView) -> TensorView:
        """
        Select the rows corresponding to the diagonal entries in the expression and sum along
        axis 0.
        """
        pass

    @abstractmethod
    def conv(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Returns view corresponding to a discrete convolution with data 'a', i.e., multiplying from
        the left a repetition of the column vector of 'a' for each column in A, shifted down one row
        after each column, i.e., a Toeplitz matrix.
        If lin_data is a row vector, we must transform the lhs to become a column vector before
        applying the convolution.

        Note: conv currently doesn't support parameters.
        """
        pass

    @abstractmethod
    def kron_r(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Returns view corresponding to Kronecker product of data 'a' with view x, i.e., kron(a,x).

        Note: kron_r currently doesn't support parameters.
        """
        pass

    @abstractmethod
    def kron_l(self, lin: LinOp, view: TensorView) -> TensorView:
        """
        Returns view corresponding to Kronecker product of view x with data 'a', i.e., kron(x,a).

        Note: kron_l currently doesn't support parameters.
        """
        pass

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

    @abstractmethod
    def get_variable_tensor(self, shape: tuple[int, ...], variable_id: int) -> Any:
        """
        Returns tensor of a variable node, i.e., eye(n) across axes 0 and 1, where n is
        the size of the variable.
        """
        pass

    @abstractmethod
    def get_data_tensor(self, data: Any) -> Any:
        """
        Returns tensor of constant node as a column vector.
        """
        pass

    @abstractmethod
    def get_param_tensor(self, shape: tuple[int, ...], parameter_id: int) -> Any:
        """
        Returns tensor of a parameter node, i.e., eye(n) across axes 0 and 2, where n is
        the size of the parameter.
        """
        pass