from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.galoistools import (
from sympy.polys.polyconfig import query
from sympy.polys.polyerrors import (

    Cancel common factors in a rational function `f/g`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_cancel(2*x**2 - 2, x**2 - 2*x + 1)
    (2*x + 2, x - 1)

    