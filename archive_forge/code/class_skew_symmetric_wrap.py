from typing import List, Tuple
import cvxpy.lin_ops.lin_op as lo
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint
class skew_symmetric_wrap(Wrap):
    """Asserts that X is a real square matrix, satisfying X + X.T == 0.
    """

    def validate_arguments(self) -> None:
        validate_real_square(self.args[0])

    def is_skew_symmetric(self) -> bool:
        return True