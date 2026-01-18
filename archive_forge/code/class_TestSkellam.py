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
class TestSkellam:

    def test_pmf(self):
        k = numpy.arange(-10, 15)
        mu1, mu2 = (10, 5)
        skpmfR = numpy.array([4.225458296192689e-05, 0.00011404838449648488, 0.0002897962580175266, 0.0006917707818210123, 0.0015480716105844708, 0.003241227496343389, 0.006337370717512329, 0.011552351566696643, 0.019606152375042644, 0.030947164083410337, 0.04540173756676736, 0.06189432816682069, 0.07842460950017058, 0.09241881253357313, 0.10139793148019728, 0.10371927988298846, 0.09907658307740609, 0.08854666007308956, 0.07418784205248681, 0.05839277286220025, 0.04326869295301316, 0.030248159818374226, 0.01999143430560302, 0.01251687730330118, 0.007438987622622971])
        assert_almost_equal(stats.skellam.pmf(k, mu1, mu2), skpmfR, decimal=15)

    def test_cdf(self):
        k = numpy.arange(-10, 15)
        mu1, mu2 = (10, 5)
        skcdfR = numpy.array([6.40614753861921e-05, 0.00017810985988267694, 0.00046790611790020336, 0.0011596768997212152, 0.0027077485103056847, 0.005948976006649072, 0.012286346724161398, 0.023838698290858034, 0.04344485066590067, 0.074392014749311, 0.11979375231607835, 0.181688080482899, 0.2601126899830695, 0.3525315025166426, 0.4539294339968399, 0.5576487138798283, 0.6567252969572344, 0.7452719570303239, 0.8194597990828106, 0.8778525719450109, 0.921121264898024, 0.9513694247163982, 0.9713608590220012, 0.9838777363253024, 0.9913167239479254])
        assert_almost_equal(stats.skellam.cdf(k, mu1, mu2), skcdfR, decimal=5)

    def test_extreme_mu2(self):
        x, mu1, mu2 = (0, 1, 4820232647677555.0)
        assert_allclose(stats.skellam.pmf(x, mu1, mu2), 0, atol=1e-16)
        assert_allclose(stats.skellam.cdf(x, mu1, mu2), 1, atol=1e-16)