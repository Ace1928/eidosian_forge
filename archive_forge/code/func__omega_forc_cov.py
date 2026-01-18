from __future__ import annotations
from statsmodels.compat.python import lrange
from collections import defaultdict
from io import StringIO
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly, deprecated_alias
from statsmodels.tools.linalg import logdet_symm
from statsmodels.tools.sm_exceptions import OutputWarning
from statsmodels.tools.validation import array_like
from statsmodels.tsa.base.tsa_model import (
import statsmodels.tsa.tsatools as tsa
from statsmodels.tsa.tsatools import duplication_matrix, unvec, vec
from statsmodels.tsa.vector_ar import output, plotting, util
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.output import VARSummary
def _omega_forc_cov(self, steps):
    G = self._zz
    Ginv = np.linalg.inv(G)
    B = self._bmat_forc_cov()
    _B = {}

    def bpow(i):
        if i not in _B:
            _B[i] = np.linalg.matrix_power(B, i)
        return _B[i]
    phis = self.ma_rep(steps)
    sig_u = self.sigma_u
    omegas = np.zeros((steps, self.neqs, self.neqs))
    for h in range(1, steps + 1):
        if h == 1:
            omegas[h - 1] = self.df_model * self.sigma_u
            continue
        om = omegas[h - 1]
        for i in range(h):
            for j in range(h):
                Bi = bpow(h - 1 - i)
                Bj = bpow(h - 1 - j)
                mult = np.trace(Bi.T @ Ginv @ Bj @ G)
                om += mult * phis[i] @ sig_u @ phis[j].T
        omegas[h - 1] = om
    return omegas