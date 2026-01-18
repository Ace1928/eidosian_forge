from __future__ import division  # Many analytical derivatives depend on this
from builtins import map
import math
import sys
import itertools
import uncertainties.core as uncert_core
from uncertainties.core import (to_affine_scalar, AffineScalarFunc,
def _deriv_pow_1(x, y):
    if x == 0 and y > 0:
        return 0.0
    else:
        return math.log(x) * math.pow(x, y)