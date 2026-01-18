from typing import List, Tuple
import cvxpy.lin_ops.lin_op as lo
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint
class hermitian_wrap(Wrap):
    """Asserts that a square matrix is Hermitian.
    """

    def validate_arguments(self) -> None:
        arg = self.args[0]
        ndim_test = len(arg.shape) == 2
        if not ndim_test:
            raise ValueError('The input must be a square matrix.')
        elif arg.shape[0] != arg.shape[1]:
            raise ValueError('The input must be a square matrix.')

    def is_hermitian(self) -> bool:
        return True