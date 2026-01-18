from typing import List, Tuple
import cvxpy.lin_ops.lin_op as lo
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint
class nonpos_wrap(Wrap):
    """Asserts that the expression is nonpositive.
    """

    def is_nonpos(self) -> bool:
        return True