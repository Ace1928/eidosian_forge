from __future__ import division
import operator as op
from functools import reduce
from typing import List, Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.conj import conj
from cvxpy.atoms.affine.reshape import deep_flatten, reshape
from cvxpy.atoms.affine.sum import sum as cvxpy_sum
from cvxpy.constraints.constraint import Constraint
from cvxpy.error import DCPError
from cvxpy.expressions.constants.parameter import (
from cvxpy.expressions.expression import Expression
class MulExpression(BinaryOperator):
    """Matrix multiplication.

    The semantics of multiplication are exactly as those of NumPy's
    matmul function, except here multiplication by a scalar is permitted.
    MulExpression objects can be created by using the '*' operator of
    the Expression class.

    Parameters
    ----------
    lh_exp : Expression
        The left-hand side of the multiplication.
    rh_exp : Expression
        The right-hand side of the multiplication.
    """
    OP_NAME = '@'
    OP_FUNC = op.mul

    def numeric(self, values):
        """Matrix multiplication.
        """
        if values[0].shape == () or values[1].shape == () or intf.is_sparse(values[0]) or intf.is_sparse(values[1]):
            return values[0] * values[1]
        else:
            return np.matmul(values[0], values[1])

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return u.shape.mul_shapes(self.args[0].shape, self.args[1].shape)

    def is_atom_convex(self) -> bool:
        """Multiplication is convex (affine) in its arguments only if one of
           the arguments is constant.
        """
        if u.scopes.dpp_scope_active():
            x = self.args[0]
            y = self.args[1]
            return (x.is_constant() or y.is_constant()) or (is_param_affine(x) and is_param_free(y)) or (is_param_affine(y) and is_param_free(x))
        else:
            return self.args[0].is_constant() or self.args[1].is_constant()

    def is_atom_concave(self) -> bool:
        """If the multiplication atom is convex, then it is affine.
        """
        return self.is_atom_convex()

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return self.args[1 - idx].is_nonneg()

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return self.args[1 - idx].is_nonpos()

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        if self.args[0].is_constant() or self.args[1].is_constant():
            return super(MulExpression, self)._grad(values)
        X = values[0]
        Y = values[1]
        DX_rows = self.args[0].size
        cols = self.args[0].size
        DX = sp.dok_matrix((DX_rows, cols))
        for k in range(self.args[0].shape[0]):
            DX[k::self.args[0].shape[0], k::self.args[0].shape[0]] = Y
        DX = sp.csc_matrix(DX)
        cols = 1 if len(self.args[1].shape) == 1 else self.args[1].shape[1]
        DY = sp.block_diag([X.T for k in range(cols)], 'csc')
        return [DX, DY]

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List[Constraint]]:
        """Multiply the linear expressions.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        lhs = arg_objs[0]
        rhs = arg_objs[1]
        if self.args[0].is_constant():
            return (lu.mul_expr(lhs, rhs, shape), [])
        elif self.args[1].is_constant():
            return (lu.rmul_expr(lhs, rhs, shape), [])
        else:
            raise DCPError('Product of two non-constant expressions is not DCP.')