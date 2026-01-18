import numpy as np
from .._shared.utils import _supported_float_type
from .colorconv import lab2lch, _cart2polar_2pi
squared hue difference term occurring in deltaE_cmc and deltaE_ciede94

    Despite its name, "dH" is not a simple difference of hue values.  We avoid
    working directly with the hue value, since differencing angles is
    troublesome.  The hue term is usually written as:
        c1 = sqrt(a1**2 + b1**2)
        c2 = sqrt(a2**2 + b2**2)
        term = (a1-a2)**2 + (b1-b2)**2 - (c1-c2)**2
        dH = sqrt(term)

    However, this has poor roundoff properties when a or b is dominant.
    Instead, ab is a vector with elements a and b.  The same dH term can be
    re-written as:
        |ab1-ab2|**2 - (|ab1| - |ab2|)**2
    and then simplified to:
        2*|ab1|*|ab2| - 2*dot(ab1, ab2)
    