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
class TestArgus:

    def test_argus_rvs_large_chi(self):
        x = stats.argus.rvs(50, size=500, random_state=325)
        assert_almost_equal(stats.argus(50).mean(), x.mean(), decimal=4)

    @pytest.mark.parametrize('chi, random_state', [[0.1, 325], [1.3, 155], [3.5, 135]])
    def test_rvs(self, chi, random_state):
        x = stats.argus.rvs(chi, size=500, random_state=random_state)
        _, p = stats.kstest(x, 'argus', (chi,))
        assert_(p > 0.05)

    @pytest.mark.parametrize('chi', [1e-09, 1e-06])
    def test_rvs_small_chi(self, chi):
        r = stats.argus.rvs(chi, size=500, random_state=890981)
        _, p = stats.kstest(r, lambda x: 1 - (1 - x ** 2) ** (3 / 2))
        assert_(p > 0.05)

    @pytest.mark.parametrize('chi, expected_mean', [(1, 0.6187026683551835), (10, 0.984805536783744), (40, 0.9990617659702923), (60, 0.99958318851653), (99, 0.9998469348663028)])
    def test_mean(self, chi, expected_mean):
        m = stats.argus.mean(chi, scale=1)
        assert_allclose(m, expected_mean, rtol=1e-13)

    @pytest.mark.parametrize('chi, expected_var, rtol', [(1, 0.05215651254197807, 1e-13), (10, 0.00015805472008165595, 1e-11), (40, 5.877763210262901e-07, 1e-08), (60, 1.1590179389611416e-07, 1e-08), (99, 1.5623277006064666e-08, 1e-08)])
    def test_var(self, chi, expected_var, rtol):
        v = stats.argus.var(chi, scale=1)
        assert_allclose(v, expected_var, rtol=rtol)

    @pytest.mark.parametrize('chi, expected, rtol', [(0.9, 0.07646314974436118, 1e-14), (0.5, 0.015429797891863365, 1e-14), (0.1, 0.0001325825293278049, 1e-14), (0.01, 1.3297677078224565e-07, 1e-15), (0.001, 1.3298072023958999e-10, 1e-14), (0.0001, 1.3298075973486862e-13, 1e-14), (1e-06, 1.32980760133771e-19, 1e-14), (1e-09, 1.329807601338109e-28, 1e-15)])
    def test_argus_phi_small_chi(self, chi, expected, rtol):
        assert_allclose(_argus_phi(chi), expected, rtol=rtol)

    @pytest.mark.parametrize('chi, expected', [(0.5, (0.28414073302940573, 1.2742227939992954, 1.2381254688255896)), (0.2, (0.296172952995264, 1.2951290588110516, 1.1865767100877576)), (0.1, (0.29791447523536274, 1.29806307956989, 1.1793168289857412)), (0.01, (0.2984904104866452, 1.2990283628160553, 1.1769268414080531)), (0.001, (0.298496172925224, 1.2990380082487925, 1.176902956021053)), (0.0001, (0.29849623054991836, 1.2990381047023793, 1.1769027171686324)), (1e-06, (0.2984962311319278, 1.2990381056765605, 1.1769027147562232)), (1e-09, (0.298496231131986, 1.299038105676658, 1.1769027147559818))])
    def test_pdf_small_chi(self, chi, expected):
        x = np.array([0.1, 0.5, 0.9])
        assert_allclose(stats.argus.pdf(x, chi), expected, rtol=1e-13)

    @pytest.mark.parametrize('chi, expected', [(0.5, (0.9857660526895221, 0.6616565930168475, 0.08796070398429937)), (0.2, (0.9851555052359501, 0.6514666238985464, 0.08362690023746594)), (0.1, (0.9850670974995661, 0.6500061310508574, 0.08302050640683846)), (0.01, (0.9850378582451867, 0.6495239242251358, 0.08282109244852445)), (0.001, (0.9850375656906663, 0.6495191015522573, 0.08281910005231098)), (0.0001, (0.9850375627651049, 0.6495190533254682, 0.08281908012852317)), (1e-06, (0.9850375627355568, 0.6495190528383777, 0.08281907992729293)), (1e-09, (0.9850375627355538, 0.649519052838329, 0.0828190799272728))])
    def test_sf_small_chi(self, chi, expected):
        x = np.array([0.1, 0.5, 0.9])
        assert_allclose(stats.argus.sf(x, chi), expected, rtol=1e-14)

    @pytest.mark.parametrize('chi, expected', [(0.5, (0.0142339473104779, 0.3383434069831524, 0.9120392960157007)), (0.2, (0.014844494764049919, 0.34853337610145363, 0.916373099762534)), (0.1, (0.014932902500433911, 0.34999386894914264, 0.9169794935931616)), (0.01, (0.014962141754813293, 0.35047607577486417, 0.9171789075514756)), (0.001, (0.01496243430933372, 0.35048089844774266, 0.917180899947689)), (0.0001, (0.014962437234895118, 0.3504809466745317, 0.9171809198714769)), (1e-06, (0.01496243726444329, 0.3504809471616223, 0.9171809200727071)), (1e-09, (0.014962437264446245, 0.350480947161671, 0.9171809200727272))])
    def test_cdf_small_chi(self, chi, expected):
        x = np.array([0.1, 0.5, 0.9])
        assert_allclose(stats.argus.cdf(x, chi), expected, rtol=1e-12)

    @pytest.mark.parametrize('chi, expected, rtol', [(0.5, (0.5964284712757741, 0.052890651988588604), 1e-12), (0.101, (0.5893490968089076, 0.053017469847275685), 1e-11), (0.1, (0.5893431757009437, 0.05301755449499372), 1e-13), (0.01, (0.5890515677940915, 0.05302167905837031), 1e-13), (0.001, (0.5890486520005177, 0.053021719862088104), 1e-13), (0.0001, (0.5890486228426105, 0.0530217202700811), 1e-13), (1e-06, (0.5890486225481156, 0.05302172027420182), 1e-13), (1e-09, (0.5890486225480862, 0.05302172027420224), 1e-13)])
    def test_stats_small_chi(self, chi, expected, rtol):
        val = stats.argus.stats(chi, moments='mv')
        assert_allclose(val, expected, rtol=rtol)