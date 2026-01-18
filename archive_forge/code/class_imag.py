from typing import Tuple
import numpy as np
from cvxpy.atoms.affine.affine_atom import AffAtom
class imag(AffAtom):
    """Extracts the imaginary part of an expression.
    """

    def __init__(self, expr) -> None:
        super(imag, self).__init__(expr)

    def numeric(self, values):
        """Convert the vector constant into a diagonal matrix.
        """
        return np.imag(values[0])

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the shape of the expression.
        """
        return self.args[0].shape

    def is_imag(self) -> bool:
        """Is the expression imaginary?
        """
        return False

    def is_complex(self) -> bool:
        """Is the expression complex valued?
        """
        return False

    def is_symmetric(self) -> bool:
        """Is the expression symmetric?
        """
        return self.args[0].is_hermitian()