from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import connected_components
from sympy.core.sympify import sympify
from sympy.core.numbers import Integer, Rational
from sympy.matrices.dense import MutableDenseMatrix
from sympy.polys.domains import ZZ, QQ
from sympy.polys.domains import EX
from sympy.polys.rings import sring
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.domainmatrix import DomainMatrix
class PolyNonlinearError(Exception):
    """Raised by solve_lin_sys for nonlinear equations"""
    pass