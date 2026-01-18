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
def _assert_less_or_close_loglike(dist, data, func=None, **kwds):
    """
    This utility function checks that the negative log-likelihood function
    (or `func`) of the result computed using dist.fit() is less than or equal
    to the result computed using the generic fit method.  Because of
    normal numerical imprecision, the "equality" check is made using
    `np.allclose` with a relative tolerance of 1e-15.
    """
    if func is None:
        func = dist.nnlf
    mle_analytical = dist.fit(data, **kwds)
    numerical_opt = super(type(dist), dist).fit(data, **kwds)
    ll_mle_analytical = func(mle_analytical, data)
    ll_numerical_opt = func(numerical_opt, data)
    assert ll_mle_analytical <= ll_numerical_opt or np.allclose(ll_mle_analytical, ll_numerical_opt, rtol=1e-15)
    if 'floc' in kwds:
        assert mle_analytical[-2] == kwds['floc']
    if 'fscale' in kwds:
        assert mle_analytical[-1] == kwds['fscale']