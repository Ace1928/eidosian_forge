from __future__ import annotations, division
import operator
from typing import List
import numpy as np
import scipy.sparse as sp
from cvxpy.cvxcore.python import canonInterface
from cvxpy.lin_ops.canon_backend import TensorRepresentation
from cvxpy.lin_ops.lin_op import NO_OP, LinOp
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.utilities.replace_quad_forms import (
class CoeffExtractor:

    def __init__(self, inverse_data, canon_backend: str | None) -> None:
        self.id_map = inverse_data.var_offsets
        self.x_length = inverse_data.x_length
        self.var_shapes = inverse_data.var_shapes
        self.param_shapes = inverse_data.param_shapes
        self.param_to_size = inverse_data.param_to_size
        self.param_id_map = inverse_data.param_id_map
        self.canon_backend = canon_backend

    def affine(self, expr):
        """Extract problem data tensor from an expression that is reducible to
        A*x + b.

        Applying the tensor to a flattened parameter vector and reshaping
        will recover A and b (see the helpers in canonInterface).

        Parameters
        ----------
        expr : Expression or list of Expressions.
            The expression(s) to process.

        Returns
        -------
        SciPy CSR matrix
            Problem data tensor, of shape
            (constraint length * (variable length + 1), parameter length + 1)
        """
        if isinstance(expr, list):
            expr_list = expr
        else:
            expr_list = [expr]
        assert all([e.is_dpp() for e in expr_list])
        num_rows = sum([e.size for e in expr_list])
        op_list = [e.canonical_form[0] for e in expr_list]
        return canonInterface.get_problem_matrix(op_list, self.x_length, self.id_map, self.param_to_size, self.param_id_map, num_rows, self.canon_backend)

    def extract_quadratic_coeffs(self, affine_expr, quad_forms):
        """ Assumes quadratic forms all have variable arguments.
            Affine expressions can be anything.
        """
        assert affine_expr.is_dpp()
        affine_id_map, affine_offsets, x_length, affine_var_shapes = InverseData.get_var_offsets(affine_expr.variables())
        op_list = [affine_expr.canonical_form[0]]
        param_coeffs = canonInterface.get_problem_matrix(op_list, x_length, affine_offsets, self.param_to_size, self.param_id_map, affine_expr.size, self.canon_backend)
        constant = param_coeffs[-1, :]
        c = param_coeffs[:-1, :].toarray()
        num_params = param_coeffs.shape[1]
        coeffs = {}
        for var in affine_expr.variables():
            if var.id in quad_forms:
                var_id = var.id
                orig_id = quad_forms[var_id][2].args[0].id
                var_offset = affine_id_map[var_id][0]
                var_size = affine_id_map[var_id][1]
                c_part = c[var_offset:var_offset + var_size, :]
                P = quad_forms[var_id][2].P
                assert P.value is not None, 'P matrix must be instantiated before calling extract_quadratic_coeffs.'
                if sp.issparse(P) and (not isinstance(P, sp.coo_matrix)):
                    P = P.value.tocoo()
                else:
                    P = sp.coo_matrix(P.value)
                if var_size == 1:
                    nonzero_idxs = c_part[0] != 0
                    data = P.data[:, None] * c_part[:, nonzero_idxs]
                    param_idxs = np.arange(num_params)[nonzero_idxs]
                    P_tup = TensorRepresentation(data.flatten(order='F'), np.tile(P.row, len(param_idxs)), np.tile(P.col, len(param_idxs)), np.repeat(param_idxs, len(P.data)), P.shape)
                else:
                    assert (P.col == P.row).all(), 'Only diagonal P matrices are supported for multiple quad forms.'
                    scaled_c_part = P @ c_part
                    paramx_idx_row, param_idx_col = np.nonzero(scaled_c_part)
                    c_vals = c_part[paramx_idx_row, param_idx_col]
                    P_tup = TensorRepresentation(c_vals, paramx_idx_row, paramx_idx_row, param_idx_col, P.shape)
                if orig_id in coeffs:
                    if 'P' in coeffs[orig_id]:
                        coeffs[orig_id]['P'] = coeffs[orig_id]['P'] + P_tup
                    else:
                        coeffs[orig_id]['P'] = P_tup
                else:
                    coeffs[orig_id] = dict()
                    coeffs[orig_id]['P'] = P_tup
                    shape = (P.shape[0], c.shape[1])
                    if num_params == 1:
                        coeffs[orig_id]['q'] = np.zeros(shape)
                    else:
                        coeffs[orig_id]['q'] = sp.coo_matrix(([], ([], [])), shape=shape)
            else:
                var_offset = affine_id_map[var.id][0]
                var_size = np.prod(affine_var_shapes[var.id], dtype=int)
                if var.id in coeffs:
                    if num_params == 1:
                        coeffs[var.id]['q'] += c[var_offset:var_offset + var_size, :]
                    else:
                        coeffs[var.id]['q'] += param_coeffs[var_offset:var_offset + var_size, :]
                else:
                    coeffs[var.id] = dict()
                    if num_params == 1:
                        coeffs[var.id]['q'] = c[var_offset:var_offset + var_size, :]
                    else:
                        coeffs[var.id]['q'] = param_coeffs[var_offset:var_offset + var_size, :]
        return (coeffs, constant)

    def quad_form(self, expr):
        """Extract quadratic, linear constant parts of a quadratic objective.
        """
        root = LinOp(NO_OP, expr.shape, [expr], [])
        quad_forms = replace_quad_forms(root, {})
        coeffs, constant = self.extract_quadratic_coeffs(root.args[0], quad_forms)
        restore_quad_forms(root.args[0], quad_forms)
        offsets = sorted(self.id_map.items(), key=operator.itemgetter(1))
        num_params = constant.shape[1]
        P_list = []
        q_list = []
        P_height = 0
        P_entries = 0
        for var_id, _ in offsets:
            shape = self.var_shapes[var_id]
            size = np.prod(shape, dtype=int)
            if var_id in coeffs and 'P' in coeffs[var_id]:
                P = coeffs[var_id]['P']
                P_entries += P.data.size
            else:
                P = TensorRepresentation.empty_with_shape((size, size))
            if var_id in coeffs and 'q' in coeffs[var_id]:
                q = coeffs[var_id]['q']
            elif num_params == 1:
                q = np.zeros((size, num_params))
            else:
                q = sp.coo_matrix(([], ([], [])), (size, num_params))
            P_list.append(P)
            q_list.append(q)
            P_height += size
        if P_height != self.x_length:
            raise ValueError('Resulting quadratic form does not have appropriate dimensions')
        P = self.merge_P_list(P_list, P_height, num_params)
        q = self.merge_q_list(q_list, constant, num_params)
        return (P, q)

    def merge_P_list(self, P_list: List[TensorRepresentation], P_height: int, num_params: int) -> sp.csc_matrix:
        """Conceptually we build a block diagonal matrix
           out of all the Ps, then flatten the first two dimensions.
           eg P1
                P2
           We do this by extending each P with zero blocks above and below.

        Args:
            P_list: list of P submatrices as TensorRepresentation objects.
            P_entries: number of entries in the merged P matrix.
            P_height: number of rows in the merged P matrix.
            num_params: number of parameters in the problem.
        
        Returns:
            A CSC sparse representation of the merged P matrix.
        """
        offset = 0
        for P in P_list:
            m, n = P.shape
            assert m == n
            P.row += offset
            P.col += offset
            P.shape = (P_height, P_height)
            offset += m
        combined = TensorRepresentation.combine(P_list)
        return combined.flatten_tensor(num_params)

    def merge_q_list(self, q_list: List[sp.spmatrix | np.ndarray], constant: sp.csc_matrix, num_params: int) -> sp.csr_matrix:
        """Stack q with constant offset as last row.

        Args:
            q_list: list of q submatrices as SciPy sparse matrices or NumPy arrays.
            constant: The constant offset as a CSC sparse matrix.
            num_params: number of parameters in the problem.

        Returns:
            A CSR sparse representation of the merged q matrix.
        """
        if num_params == 1:
            q = np.vstack(q_list)
            q = np.vstack([q, constant.A])
            return sp.csr_matrix(q)
        else:
            q = sp.vstack(q_list + [constant])
            return sp.csr_matrix(q)