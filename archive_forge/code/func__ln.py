from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.relational import is_eq
from sympy.functions.elementary.complexes import (conjugate, im, re, sign)
from sympy.functions.elementary.exponential import (exp, log as ln)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan2)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.simplify.trigsimp import trigsimp
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.core.sympify import sympify, _sympify
from sympy.core.expr import Expr
from sympy.core.logic import fuzzy_not, fuzzy_or
from mpmath.libmp.libmpf import prec_to_dps
def _ln(self):
    """Returns the natural logarithm of the quaternion (_ln(q)).

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q._ln()
        log(sqrt(30))
        + 2*sqrt(29)*acos(sqrt(30)/30)/29*i
        + 3*sqrt(29)*acos(sqrt(30)/30)/29*j
        + 4*sqrt(29)*acos(sqrt(30)/30)/29*k

        """
    q = self
    vector_norm = sqrt(q.b ** 2 + q.c ** 2 + q.d ** 2)
    q_norm = q.norm()
    a = ln(q_norm)
    b = q.b * acos(q.a / q_norm) / vector_norm
    c = q.c * acos(q.a / q_norm) / vector_norm
    d = q.d * acos(q.a / q_norm) / vector_norm
    return Quaternion(a, b, c, d)