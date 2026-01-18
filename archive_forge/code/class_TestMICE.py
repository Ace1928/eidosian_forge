import numpy as np
import pandas as pd
import pytest
from statsmodels.imputation import mice
import statsmodels.api as sm
from numpy.testing import assert_equal, assert_allclose
import warnings
class TestMICE:

    def test_MICE(self):
        df = gendat()
        imp_data = mice.MICEData(df)
        mi = mice.MICE('y ~ x1 + x2 + x1:x2', sm.OLS, imp_data)
        result = mi.fit(1, 3)
        assert issubclass(result.__class__, mice.MICEResults)
        smr = result.summary()

    def test_MICE1(self):
        df = gendat()
        imp_data = mice.MICEData(df)
        mi = mice.MICE('y ~ x1 + x2 + x1:x2', sm.OLS, imp_data)
        from statsmodels.regression.linear_model import RegressionResultsWrapper
        for j in range(3):
            x = mi.next_sample()
            assert issubclass(x.__class__, RegressionResultsWrapper)

    def test_MICE1_regularized(self):
        df = gendat()
        imp = mice.MICEData(df, perturbation_method='boot')
        imp.set_imputer('x1', 'x2 + y', fit_kwds={'alpha': 1, 'L1_wt': 0})
        imp.update_all()

    def test_MICE2(self):
        from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper
        df = gendat()
        imp_data = mice.MICEData(df)
        mi = mice.MICE('x3 ~ x1 + x2', sm.GLM, imp_data, init_kwds={'family': sm.families.Binomial()})
        for j in range(3):
            x = mi.next_sample()
            assert isinstance(x, GLMResultsWrapper)
            assert isinstance(x.family, sm.families.Binomial)

    @pytest.mark.slow
    def t_est_combine(self):
        gen = np.random.RandomState(3897)
        x1 = gen.normal(size=300)
        x2 = gen.normal(size=300)
        y = x1 + x2 + gen.normal(size=300)
        x1[0:100] = np.nan
        x2[250:] = np.nan
        df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
        idata = mice.MICEData(df)
        mi = mice.MICE('y ~ x1 + x2', sm.OLS, idata, n_skip=20)
        result = mi.fit(10, 20)
        fmi = np.asarray([0.1778143, 0.11057262, 0.29626521])
        assert_allclose(result.frac_miss_info, fmi, atol=1e-05)
        params = np.asarray([-0.03486102, 0.96236808, 0.9970371])
        assert_allclose(result.params, params, atol=1e-05)
        tvalues = np.asarray([-0.54674776, 15.28091069, 13.61359403])
        assert_allclose(result.tvalues, tvalues, atol=1e-05)