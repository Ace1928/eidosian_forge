import numpy as np
from statsmodels.datasets.ccard.data import load_pandas
from statsmodels.stats.oaxaca import OaxacaBlinder
from statsmodels.tools.tools import add_constant
class TestOneModel:

    @classmethod
    def setup_class(cls):
        np.random.seed(0)
        cls.one_model = OaxacaBlinder(pandas_df.endog, pandas_df.exog, 'OWNRENT', hasconst=False).two_fold(True, two_fold_type='self_submitted', submitted_weight=1)

    def test_results(self):
        unexp, exp, gap = self.one_model.params
        unexp_std, exp_std = self.one_model.std
        one_params_stata_results = np.array([75.4537, 83.29673, 158.75044])
        one_std_stata_results = np.array([64.58479, 71.05619])
        np.testing.assert_almost_equal(unexp, one_params_stata_results[0], 3)
        np.testing.assert_almost_equal(exp, one_params_stata_results[1], 3)
        np.testing.assert_almost_equal(gap, one_params_stata_results[2], 3)
        np.testing.assert_almost_equal(unexp_std, one_std_stata_results[0], 3)
        np.testing.assert_almost_equal(exp_std, one_std_stata_results[1], 3)