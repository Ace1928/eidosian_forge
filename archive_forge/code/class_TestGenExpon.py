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
class TestGenExpon:

    def test_pdf_unity_area(self):
        from scipy.integrate import simpson
        p = stats.genexpon.pdf(numpy.arange(0, 10, 0.01), 0.5, 0.5, 2.0)
        assert_almost_equal(simpson(p, dx=0.01), 1, 1)

    def test_cdf_bounds(self):
        cdf = stats.genexpon.cdf(numpy.arange(0, 10, 0.01), 0.5, 0.5, 2.0)
        assert_(numpy.all((0 <= cdf) & (cdf <= 1)))

    @pytest.mark.parametrize('x, p, a, b, c', [(15, 1.0859444834514553e-19, 1, 2, 1.5), (0.25, 0.7609068232534623, 0.5, 2, 3), (0.25, 0.09026661397565876, 9.5, 2, 0.5), (0.01, 0.9753038265071597, 2.5, 0.25, 0.5), (3.25, 0.0001962824553094492, 2.5, 0.25, 0.5), (0.125, 0.9508674287164001, 0.25, 5, 0.5)])
    def test_sf_isf(self, x, p, a, b, c):
        sf = stats.genexpon.sf(x, a, b, c)
        assert_allclose(sf, p, rtol=2e-14)
        isf = stats.genexpon.isf(p, a, b, c)
        assert_allclose(isf, x, rtol=2e-14)

    @pytest.mark.parametrize('x, p, a, b, c', [(0.25, 0.2390931767465377, 0.5, 2, 3), (0.25, 0.9097333860243412, 9.5, 2, 0.5), (0.01, 0.0246961734928403, 2.5, 0.25, 0.5), (3.25, 0.9998037175446906, 2.5, 0.25, 0.5), (0.125, 0.04913257128359998, 0.25, 5, 0.5)])
    def test_cdf_ppf(self, x, p, a, b, c):
        cdf = stats.genexpon.cdf(x, a, b, c)
        assert_allclose(cdf, p, rtol=2e-14)
        ppf = stats.genexpon.ppf(p, a, b, c)
        assert_allclose(ppf, x, rtol=2e-14)