import numpy as np
from numpy.testing import assert_, assert_allclose
from numpy import pi
import pytest
import itertools
from scipy._lib import _pep440
import scipy.special as sc
from scipy.special._testutils import (
from scipy.special._mptestutils import (
from scipy.special._ufuncs import (
def _mpmath_wrightomega(z, dps):
    with mpmath.workdps(dps):
        z = mpmath.mpc(z)
        unwind = mpmath.ceil((z.imag - mpmath.pi) / (2 * mpmath.pi))
        res = mpmath.lambertw(mpmath.exp(z), unwind)
    return res