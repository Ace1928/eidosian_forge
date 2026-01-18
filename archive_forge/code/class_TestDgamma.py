import warnings
import re
import sys
import pickle
from pathlib import Path
import os
import json
import platform
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import numpy
import numpy as np
from numpy import typecodes, array
from numpy.lib.recfunctions import rec_append_fields
from scipy import special
from scipy._lib._util import check_random_state
from scipy.integrate import (IntegrationWarning, quad, trapezoid,
import scipy.stats as stats
from scipy.stats._distn_infrastructure import argsreduce
import scipy.stats.distributions
from scipy.special import xlogy, polygamma, entr
from scipy.stats._distr_params import distcont, invdistcont
from .test_discrete_basic import distdiscrete, invdistdiscrete
from scipy.stats._continuous_distns import FitDataError, _argus_phi
from scipy.optimize import root, fmin, differential_evolution
from itertools import product
class TestDgamma:

    def test_pdf(self):
        rng = np.random.default_rng(3791303244302340058)
        size = 10
        x = rng.normal(scale=10, size=size)
        a = rng.uniform(high=10, size=size)
        res = stats.dgamma.pdf(x, a)
        ref = stats.gamma.pdf(np.abs(x), a) / 2
        assert_allclose(res, ref)
        dist = stats.dgamma(a)
        assert_allclose(dist.pdf(x), res, rtol=5e-16)

    @pytest.mark.parametrize('x, a, expected', [(-20, 1, 1.030576811219279e-09), (-40, 1, 2.1241771276457944e-18), (-50, 5, 2.7248509914602648e-17), (-25, 0.125, 5.333071920958156e-14), (5, 1, 0.9966310265004573)])
    def test_cdf_ppf_sf_isf_tail(self, x, a, expected):
        cdf = stats.dgamma.cdf(x, a)
        assert_allclose(cdf, expected, rtol=5e-15)
        ppf = stats.dgamma.ppf(expected, a)
        assert_allclose(ppf, x, rtol=5e-15)
        sf = stats.dgamma.sf(-x, a)
        assert_allclose(sf, expected, rtol=5e-15)
        isf = stats.dgamma.isf(expected, a)
        assert_allclose(isf, -x, rtol=5e-15)

    @pytest.mark.parametrize('a, ref', [(1.5, 2.0541199559354117), (1.3, 1.9357296377121247), (1.1, 1.7856502333412134)])
    def test_entropy(self, a, ref):
        assert_allclose(stats.dgamma.entropy(a), ref, rtol=1e-14)

    @pytest.mark.parametrize('a, ref', [(1e-100, -1e+100), (1e-10, -9999999975.858217), (1e-05, -99987.37111657023), (10000.0, 6.717222565586032), (1000000000000000.0, 19.38147391121996), (1e+100, 117.2413403634669)])
    def test_entropy_entreme_values(self, a, ref):
        assert_allclose(stats.dgamma.entropy(a), ref, rtol=1e-10)

    def test_entropy_array_input(self):
        x = np.array([1, 5, 1e+20, 1e-05])
        y = stats.dgamma.entropy(x)
        for i in range(len(y)):
            assert y[i] == stats.dgamma.entropy(x[i])