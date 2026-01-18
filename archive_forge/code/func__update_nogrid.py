from statsmodels.compat.pandas import Appender
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from scipy import linalg as spl
from statsmodels.stats.correlation_tools import cov_nearest
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import bool_like
def _update_nogrid(self, params):
    endog = self.model.endog_li
    time = self.model.time_li
    if self.designx is not None:
        designx = self.designx
    else:
        designx = []
        for i in range(self.model.num_group):
            ngrp = len(endog[i])
            if ngrp == 0:
                continue
            for j1 in range(ngrp):
                for j2 in range(j1):
                    designx.append(self.dist_func(time[i][j1, :], time[i][j2, :]))
        designx = np.array(designx)
        self.designx = designx
    scale = self.model.estimate_scale()
    varfunc = self.model.family.variance
    cached_means = self.model.cached_means
    var = 1.0 - self.dep_params ** (2 * designx)
    var /= 1.0 - self.dep_params ** 2
    wts = 1.0 / var
    wts /= wts.sum()
    residmat = []
    for i in range(self.model.num_group):
        expval, _ = cached_means[i]
        stdev = np.sqrt(scale * varfunc(expval))
        resid = (endog[i] - expval) / stdev
        ngrp = len(resid)
        for j1 in range(ngrp):
            for j2 in range(j1):
                residmat.append([resid[j1], resid[j2]])
    residmat = np.array(residmat)

    def fitfunc(a):
        dif = residmat[:, 0] - a ** designx * residmat[:, 1]
        return np.dot(dif ** 2, wts)
    b_lft, f_lft = (0.0, fitfunc(0.0))
    b_ctr, f_ctr = (0.5, fitfunc(0.5))
    while f_ctr > f_lft:
        b_ctr /= 2
        f_ctr = fitfunc(b_ctr)
        if b_ctr < 1e-08:
            self.dep_params = 0
            return
    b_rgt, f_rgt = (0.75, fitfunc(0.75))
    while f_rgt < f_ctr:
        b_rgt = b_rgt + (1.0 - b_rgt) / 2
        f_rgt = fitfunc(b_rgt)
        if b_rgt > 1.0 - 1e-06:
            raise ValueError('Autoregressive: unable to find right bracket')
    from scipy.optimize import brent
    self.dep_params = brent(fitfunc, brack=[b_lft, b_ctr, b_rgt])