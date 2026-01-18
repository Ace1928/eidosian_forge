from __future__ import division  # Many analytical derivatives depend on this
from builtins import map
import math
import sys
import itertools
import uncertainties.core as uncert_core
from uncertainties.core import (to_affine_scalar, AffineScalarFunc,
def _deriv_pow_0(x, y):
    if y == 0:
        return 0.0
    elif x != 0 or y % 1 == 0:
        return y * math.pow(x, y - 1)
    else:
        return float('nan')