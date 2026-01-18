import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.discrete.discrete_model import Poisson
import statsmodels.discrete._diagnostics_count as dia
from statsmodels.discrete.diagnostic import PoissonDiagnostic
class TestPoissonDiagnosticClass:

    @classmethod
    def setup_class(cls):
        np.random.seed(987125643)
        nr = 1
        n_groups = 2
        labels = np.arange(n_groups)
        x = np.repeat(labels, np.array([40, 60]) * nr)
        nobs = x.shape[0]
        exog = (x[:, None] == labels).astype(np.float64)
        beta = np.array([0.1, 0.3], np.float64)
        linpred = exog @ beta
        mean = np.exp(linpred)
        y = np.random.poisson(mean)
        cls.endog = y
        cls.exog = exog

    def test_spec_tests(self):
        res_dispersion = np.array([[0.1396096387543, 0.8889684245877], [0.1396096387543, 0.8889684245877], [0.2977840351238, 0.7658680002106], [0.1307899995877, 0.8959414342111], [0.1307899995877, 0.8959414342111], [0.1357101381056, 0.8920504328246], [0.2776587511235, 0.7812743277372]])
        res_zi = np.array([[0.1389582826821, 0.7093188241734], [-0.3727710861669, 0.7093188241734], [-0.2496729648642, 0.8028402670888], [0.0601651553909, 0.806235095888]])
        respoi = Poisson(self.endog, self.exog).fit(disp=0)
        dia = PoissonDiagnostic(respoi)
        t_disp = dia.test_dispersion()
        res_disp = np.column_stack((t_disp.statistic, t_disp.pvalue))
        assert_allclose(res_disp, res_dispersion, rtol=1e-08)
        nobs = self.endog.shape[0]
        t_zi_jh = dia.test_poisson_zeroinflation(method='broek', exog_infl=np.ones(nobs))
        t_zib = dia.test_poisson_zeroinflation(method='broek')
        t_zim = dia.test_poisson_zeroinflation(method='prob')
        t_zichi2 = dia.test_chisquare_prob(bin_edges=np.arange(3))
        t_zi = np.vstack([t_zi_jh[:2], t_zib[:2], t_zim[:2], t_zichi2[:2]])
        assert_allclose(t_zi, res_zi, rtol=1e-08)
        t_zi_ex = dia.test_poisson_zeroinflation(method='broek', exog_infl=self.exog)
        res_zi_ex = np.array([3.7813218150779, 0.1509719973257])
        assert_allclose(t_zi_ex[:2], res_zi_ex, rtol=1e-08)