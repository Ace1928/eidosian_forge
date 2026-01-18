import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
import scipy.stats as stats
from statsmodels.discrete.discrete_model import Logit
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.sm_exceptions import HessianInversionWarning
from statsmodels.tools.tools import add_constant
from .results.results_ordinal_model import data_store as ds
class TestCLogLogModel(CheckOrdinalModelMixin):

    @classmethod
    def setup_class(cls):
        data = ds.df
        data_unordered = ds.df_unordered

        class CLogLog(stats.rv_continuous):

            def _ppf(self, q):
                return np.log(-np.log(1 - q))

            def _cdf(self, x):
                return 1 - np.exp(-np.exp(x))
        cloglog = CLogLog()
        mod = OrderedModel(data['apply'].values.codes, np.asarray(data[['pared', 'public', 'gpa']], float), distr=cloglog)
        res = mod.fit(method='bfgs', disp=False)
        modp = OrderedModel(data['apply'], data[['pared', 'public', 'gpa']], distr=cloglog)
        resp = modp.fit(method='bfgs', disp=False)
        modf = OrderedModel.from_formula('apply ~ pared + public + gpa - 1', data={'apply': data['apply'].values.codes, 'pared': data['pared'], 'public': data['public'], 'gpa': data['gpa']}, distr=cloglog)
        resf = modf.fit(method='bfgs', disp=False)
        modu = OrderedModel(data_unordered['apply'].values.codes, np.asarray(data_unordered[['pared', 'public', 'gpa']], float), distr=cloglog)
        resu = modu.fit(method='bfgs', disp=False)
        from .results.results_ordinal_model import res_ord_cloglog as res2
        cls.res2 = res2
        cls.res1 = res
        cls.resp = resp
        cls.resf = resf
        cls.resu = resu