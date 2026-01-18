from typing import List, Tuple
import numpy as np
from scipy.sparse import csc_matrix
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vec import vec
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
class upper_tri(AffAtom):
    """The vectorized strictly upper-triagonal entries.

    The vectorization is performed by concatenating (partial) rows.
    For example, if

    ::

        A = np.array([[10, 11, 12, 13],
                      [14, 15, 16, 17],
                      [18, 19, 20, 21],
                      [22, 23, 24, 25]])

    then we have

    ::

        upper_tri(A).value == np.array([11, 12, 13, 16, 17, 21])

    """

    def __init__(self, expr) -> None:
        super(upper_tri, self).__init__(expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Vectorize the upper triagonal entries.
        """
        value = np.zeros(self.shape[0])
        count = 0
        for i in range(values[0].shape[0]):
            for j in range(values[0].shape[1]):
                if i < j:
                    value[count] = values[0][i, j]
                    count += 1
        return value

    def validate_arguments(self) -> None:
        """Checks that the argument is a square matrix.
        """
        if not self.args[0].ndim == 2 or self.args[0].shape[0] != self.args[0].shape[1]:
            raise ValueError('Argument to upper_tri must be a square matrix.')

    def shape_from_args(self) -> Tuple[int, int]:
        """A vector.
        """
        rows, cols = self.args[0].shape
        return (rows * (cols - 1) // 2, 1)

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List[Constraint]]:
        """Vectorized strictly upper triagonal entries.

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
        return (lu.upper_tri(arg_objs[0]), [])