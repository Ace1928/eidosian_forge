from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def _mobius_to_interval(M, field):
    """Convert a Mobius transform to an open interval. """
    a, b, c, d = M
    s, t = (field(a, c), field(b, d))
    if s <= t:
        return (s, t)
    else:
        return (t, s)