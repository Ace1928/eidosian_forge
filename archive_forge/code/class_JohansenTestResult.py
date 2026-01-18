from collections import defaultdict
import numpy as np
from numpy import hstack, vstack
from numpy.linalg import inv, svd
import scipy
import scipy.stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.validation import string_like
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.coint_tables import c_sja, c_sjt
from statsmodels.tsa.tsatools import duplication_matrix, lagmat, vec
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
import statsmodels.tsa.vector_ar.irf as irf
import statsmodels.tsa.vector_ar.plotting as plot
from statsmodels.tsa.vector_ar.util import get_index, seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import (
class JohansenTestResult:
    """
    Results class for Johansen's cointegration test

    Notes
    -----
    See p. 292 in [1]_ for r0t and rkt

    References
    ----------
    .. [1] Lütkepohl, H. 2005. New Introduction to Multiple Time Series
        Analysis. Springer.
    """

    def __init__(self, rkt, r0t, eig, evec, lr1, lr2, cvt, cvm, ind):
        self._meth = 'johansen'
        self._rkt = rkt
        self._r0t = r0t
        self._eig = eig
        self._evec = evec
        self._lr1 = lr1
        self._lr2 = lr2
        self._cvt = cvt
        self._cvm = cvm
        self._ind = ind

    @property
    def rkt(self):
        """Residuals for :math:`Y_{-1}`"""
        return self._rkt

    @property
    def r0t(self):
        """Residuals for :math:`\\Delta Y`."""
        return self._r0t

    @property
    def eig(self):
        """Eigenvalues of VECM coefficient matrix"""
        return self._eig

    @property
    def evec(self):
        """Eigenvectors of VECM coefficient matrix"""
        return self._evec

    @property
    def trace_stat(self):
        """Trace statistic"""
        return self._lr1

    @property
    def lr1(self):
        """Trace statistic"""
        return self._lr1

    @property
    def max_eig_stat(self):
        """Maximum eigenvalue statistic"""
        return self._lr2

    @property
    def lr2(self):
        """Maximum eigenvalue statistic"""
        return self._lr2

    @property
    def trace_stat_crit_vals(self):
        """Critical values (90%, 95%, 99%) of trace statistic"""
        return self._cvt

    @property
    def cvt(self):
        """Critical values (90%, 95%, 99%) of trace statistic"""
        return self._cvt

    @property
    def cvm(self):
        """Critical values (90%, 95%, 99%) of maximum eigenvalue statistic."""
        return self._cvm

    @property
    def max_eig_stat_crit_vals(self):
        """Critical values (90%, 95%, 99%) of maximum eigenvalue statistic."""
        return self._cvm

    @property
    def ind(self):
        """Order of eigenvalues"""
        return self._ind

    @property
    def meth(self):
        """Test method"""
        return self._meth