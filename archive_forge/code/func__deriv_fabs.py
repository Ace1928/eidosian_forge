from __future__ import division  # Many analytical derivatives depend on this
from builtins import map
import math
import sys
import itertools
import uncertainties.core as uncert_core
from uncertainties.core import (to_affine_scalar, AffineScalarFunc,
def _deriv_fabs(x):
    if x >= 0:
        return 1
    else:
        return -1