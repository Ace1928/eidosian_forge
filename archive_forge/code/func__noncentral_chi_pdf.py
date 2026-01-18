import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
import scipy.special as sp
from scipy.special._testutils import (
from scipy.special._mptestutils import (
def _noncentral_chi_pdf(t, df, nc):
    res = mpmath.besseli(df / 2 - 1, mpmath.sqrt(nc * t))
    res *= mpmath.exp(-(t + nc) / 2) * (t / nc) ** (df / 4 - 1 / 2) / 2
    return res