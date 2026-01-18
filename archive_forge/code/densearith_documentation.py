from sympy.polys.densebasic import (
from sympy.polys.polyerrors import (ExactQuotientFailed, PolynomialDivisionFailed)

    Multiply together several polynomials in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_expand([x**2 + y**2, x + 1])
    x**3 + x**2 + x*y**2 + y**2

    