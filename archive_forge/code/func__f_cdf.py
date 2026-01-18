import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
import scipy.special as sp
from scipy.special._testutils import (
from scipy.special._mptestutils import (
def _f_cdf(dfn, dfd, x):
    if x < 0:
        return mpmath.mpf(0)
    dfn, dfd, x = (mpmath.mpf(dfn), mpmath.mpf(dfd), mpmath.mpf(x))
    ub = dfn * x / (dfn * x + dfd)
    res = mpmath.betainc(dfn / 2, dfd / 2, x2=ub, regularized=True)
    return res