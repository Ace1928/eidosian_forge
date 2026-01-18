import itertools
import sys
import pytest
import numpy as np
from numpy.testing import assert_
from scipy.special._testutils import FuncData
from scipy.special import kolmogorov, kolmogi, smirnov, smirnovi
from scipy.special._ufuncs import (_kolmogc, _kolmogci, _kolmogp,
class TestKolmogp:

    def test_nan(self):
        assert_(np.isnan(_kolmogp(np.nan)))

    def test_basic(self):
        dataset = [(0.0, -0.0), (0.2, -1.532420541338916e-10), (0.4, -0.1012254419260496), (0.6, -1.324123244249925), (0.8, -1.627024345636592), (1.0, -1.071948558356941), (1.2, -0.538512430720529), (1.4, -0.2222133182429472), (1.6, -0.07649302775520538), (1.8, -0.02208687346347873), (2.0, -0.005367402045629683)]
        dataset = np.asarray(dataset)
        FuncData(_kolmogp, dataset, (0,), 1, rtol=_rtol).check()