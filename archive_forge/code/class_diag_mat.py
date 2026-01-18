from __future__ import annotations
from typing import List, Tuple, Union
import numpy as np
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.vec import vec
from cvxpy.constraints.constraint import Constraint
class diag_mat(AffAtom):
    """Extracts the diagonal from a square matrix.
    """

    def __init__(self, expr, k: int=0) -> None:
        self.k = k
        super(diag_mat, self).__init__(expr)

    def get_data(self) -> list[int]:
        return [self.k]

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Extract the diagonal from a square matrix constant.
        """
        return np.diag(values[0], k=self.k)

    def shape_from_args(self) -> Tuple[int]:
        """A column vector.
        """
        rows, _ = self.args[0].shape
        rows -= abs(self.k)
        return (rows,)

    def is_nonneg(self) -> bool:
        """Is the expression nonnegative?
        """
        return (self.args[0].is_nonneg() or self.args[0].is_psd()) and self.k == 0

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List[Constraint]]:
        """Extracts the diagonal of a matrix.

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
        return (lu.diag_mat(arg_objs[0], self.k), [])