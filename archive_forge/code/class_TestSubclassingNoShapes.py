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
class TestSubclassingNoShapes:

    def test_only__pdf(self):
        dummy_distr = _distr_gen(name='dummy')
        assert_equal(dummy_distr.pdf(1, a=1), 42)

    def test_only__cdf(self):
        dummy_distr = _distr2_gen(name='dummy')
        assert_almost_equal(dummy_distr.pdf(1, a=1), 1)

    @pytest.mark.skipif(DOCSTRINGS_STRIPPED, reason='docstring stripped')
    def test_signature_inspection(self):
        dummy_distr = _distr_gen(name='dummy')
        assert_equal(dummy_distr.numargs, 1)
        assert_equal(dummy_distr.shapes, 'a')
        res = re.findall('logpdf\\(x, a, loc=0, scale=1\\)', dummy_distr.__doc__)
        assert_(len(res) == 1)

    @pytest.mark.skipif(DOCSTRINGS_STRIPPED, reason='docstring stripped')
    def test_signature_inspection_2args(self):
        dummy_distr = _distr6_gen(name='dummy')
        assert_equal(dummy_distr.numargs, 2)
        assert_equal(dummy_distr.shapes, 'a, b')
        res = re.findall('logpdf\\(x, a, b, loc=0, scale=1\\)', dummy_distr.__doc__)
        assert_(len(res) == 1)

    def test_signature_inspection_2args_incorrect_shapes(self):
        assert_raises(TypeError, _distr3_gen, name='dummy')

    def test_defaults_raise(self):

        class _dist_gen(stats.rv_continuous):

            def _pdf(self, x, a=42):
                return 42
        assert_raises(TypeError, _dist_gen, **dict(name='dummy'))

    def test_starargs_raise(self):

        class _dist_gen(stats.rv_continuous):

            def _pdf(self, x, a, *args):
                return 42
        assert_raises(TypeError, _dist_gen, **dict(name='dummy'))

    def test_kwargs_raise(self):

        class _dist_gen(stats.rv_continuous):

            def _pdf(self, x, a, **kwargs):
                return 42
        assert_raises(TypeError, _dist_gen, **dict(name='dummy'))