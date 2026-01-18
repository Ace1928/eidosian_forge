import numpy as np
from statsmodels.datasets.ccard.data import load_pandas
from statsmodels.stats.oaxaca import OaxacaBlinder
from statsmodels.tools.tools import add_constant
class TestPooledModel:

    @classmethod
    def setup_class(cls):
        np.random.seed(0)
        cls.pooled_model = OaxacaBlinder(pandas_df.endog, pandas_df.exog, 'OWNRENT', hasconst=False).two_fold(True)

    def test_results(self):
        unexp, exp, gap = self.pooled_model.params
        unexp_std, exp_std = self.pooled_model.std
        pool_params_stata_results = np.array([27.940908, 130.809536, 158.75044])
        pool_std_stata_results = np.array([89.209487, 58.612367])
        np.testing.assert_almost_equal(unexp, pool_params_stata_results[0], 3)
        np.testing.assert_almost_equal(exp, pool_params_stata_results[1], 3)
        np.testing.assert_almost_equal(gap, pool_params_stata_results[2], 3)
        np.testing.assert_almost_equal(unexp_std, pool_std_stata_results[0], 3)
        np.testing.assert_almost_equal(exp_std, pool_std_stata_results[1], 3)