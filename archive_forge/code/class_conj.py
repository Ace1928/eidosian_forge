from typing import List, Tuple
import numpy as np
import cvxpy.lin_ops.lin_op as lo
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint
class conj(AffAtom):
    """Complex conjugate.
    """

    def __init__(self, expr) -> None:
        super(conj, self).__init__(expr)

    def numeric(self, values):
        """Convert the vector constant into a diagonal matrix.
        """
        return np.conj(values[0])

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the shape of the expression.
        """
        return self.args[0].shape

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def is_symmetric(self) -> bool:
        """Is the expression symmetric?
        """
        return self.args[0].is_symmetric()

    def is_hermitian(self) -> bool:
        """Is the expression Hermitian?
        """
        return self.args[0].is_hermitian()

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
        return (arg_objs[0], [])