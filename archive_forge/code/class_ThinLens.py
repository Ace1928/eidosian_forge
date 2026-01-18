from sympy.core.expr import Expr
from sympy.core.numbers import (I, pi)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import atan2
from sympy.matrices.dense import Matrix, MutableDenseMatrix
from sympy.polys.rationaltools import together
from sympy.utilities.misc import filldedent
class ThinLens(RayTransferMatrix):
    """
    Ray Transfer Matrix for a thin lens.

    Parameters
    ==========

    f :
        The focal distance.

    See Also
    ========

    RayTransferMatrix

    Examples
    ========

    >>> from sympy.physics.optics import ThinLens
    >>> from sympy import symbols
    >>> f = symbols('f')
    >>> ThinLens(f)
    Matrix([
    [   1, 0],
    [-1/f, 1]])
    """

    def __new__(cls, f):
        f = sympify(f)
        return RayTransferMatrix.__new__(cls, 1, 0, -1 / f, 1)