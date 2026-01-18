import itertools
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pytest
import scipy.special as sp
from scipy.special._testutils import (
from scipy.special._mptestutils import (
class _CDFData:

    def __init__(self, spfunc, mpfunc, index, argspec, spfunc_first=True, dps=20, n=5000, rtol=None, atol=None, endpt_rtol=None, endpt_atol=None):
        self.spfunc = spfunc
        self.mpfunc = mpfunc
        self.index = index
        self.argspec = argspec
        self.spfunc_first = spfunc_first
        self.dps = dps
        self.n = n
        self.rtol = rtol
        self.atol = atol
        if not isinstance(argspec, list):
            self.endpt_rtol = None
            self.endpt_atol = None
        elif endpt_rtol is not None or endpt_atol is not None:
            if isinstance(endpt_rtol, list):
                self.endpt_rtol = endpt_rtol
            else:
                self.endpt_rtol = [endpt_rtol] * len(self.argspec)
            if isinstance(endpt_atol, list):
                self.endpt_atol = endpt_atol
            else:
                self.endpt_atol = [endpt_atol] * len(self.argspec)
        else:
            self.endpt_rtol = None
            self.endpt_atol = None

    def idmap(self, *args):
        if self.spfunc_first:
            res = self.spfunc(*args)
            if np.isnan(res):
                return np.nan
            args = list(args)
            args[self.index] = res
            with mpmath.workdps(self.dps):
                res = self.mpfunc(*tuple(args))
                res = mpf2float(res.real)
        else:
            with mpmath.workdps(self.dps):
                res = self.mpfunc(*args)
                res = mpf2float(res.real)
            args = list(args)
            args[self.index] = res
            res = self.spfunc(*tuple(args))
        return res

    def get_param_filter(self):
        if self.endpt_rtol is None and self.endpt_atol is None:
            return None
        filters = []
        for rtol, atol, spec in zip(self.endpt_rtol, self.endpt_atol, self.argspec):
            if rtol is None and atol is None:
                filters.append(None)
                continue
            elif rtol is None:
                rtol = 0.0
            elif atol is None:
                atol = 0.0
            filters.append(EndpointFilter(spec.a, spec.b, rtol, atol))
        return filters

    def check(self):
        args = get_args(self.argspec, self.n)
        param_filter = self.get_param_filter()
        param_columns = tuple(range(args.shape[1]))
        result_columns = args.shape[1]
        args = np.hstack((args, args[:, self.index].reshape(args.shape[0], 1)))
        FuncData(self.idmap, args, param_columns=param_columns, result_columns=result_columns, rtol=self.rtol, atol=self.atol, vectorized=False, param_filter=param_filter).check()