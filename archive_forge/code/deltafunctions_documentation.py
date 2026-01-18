from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.functions import DiracDelta, Heaviside
from .integrals import Integral, integrate

    deltaintegrate(f, x)

    Explanation
    ===========

    The idea for integration is the following:

    - If we are dealing with a DiracDelta expression, i.e. DiracDelta(g(x)),
      we try to simplify it.

      If we could simplify it, then we integrate the resulting expression.
      We already know we can integrate a simplified expression, because only
      simple DiracDelta expressions are involved.

      If we couldn't simplify it, there are two cases:

      1) The expression is a simple expression: we return the integral,
         taking care if we are dealing with a Derivative or with a proper
         DiracDelta.

      2) The expression is not simple (i.e. DiracDelta(cos(x))): we can do
         nothing at all.

    - If the node is a multiplication node having a DiracDelta term:

      First we expand it.

      If the expansion did work, then we try to integrate the expansion.

      If not, we try to extract a simple DiracDelta term, then we have two
      cases:

      1) We have a simple DiracDelta term, so we return the integral.

      2) We didn't have a simple term, but we do have an expression with
         simplified DiracDelta terms, so we integrate this expression.

    Examples
    ========

        >>> from sympy.abc import x, y, z
        >>> from sympy.integrals.deltafunctions import deltaintegrate
        >>> from sympy import sin, cos, DiracDelta
        >>> deltaintegrate(x*sin(x)*cos(x)*DiracDelta(x - 1), x)
        sin(1)*cos(1)*Heaviside(x - 1)
        >>> deltaintegrate(y**2*DiracDelta(x - z)*DiracDelta(y - z), y)
        z**2*DiracDelta(x - z)*Heaviside(y - z)

    See Also
    ========

    sympy.functions.special.delta_functions.DiracDelta
    sympy.integrals.integrals.Integral
    