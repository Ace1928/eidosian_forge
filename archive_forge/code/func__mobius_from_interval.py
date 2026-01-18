from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def _mobius_from_interval(I, field):
    """Convert an open interval to a Mobius transform. """
    s, t = I
    a, c = (field.numer(s), field.denom(s))
    b, d = (field.numer(t), field.denom(t))
    return (a, b, c, d)