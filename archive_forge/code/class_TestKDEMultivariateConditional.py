import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
class TestKDEMultivariateConditional(KDETestBase):

    @pytest.mark.slow
    def test_mixeddata_CV_LS(self):
        dens_ls = nparam.KDEMultivariateConditional(endog=[self.Italy_gdp], exog=[self.Italy_year], dep_type='c', indep_type='o', bw='cv_ls')
        npt.assert_allclose(dens_ls.bw, [1.01203728, 0.31905144], atol=1e-05)

    def test_continuous_CV_ML(self):
        dens_ml = nparam.KDEMultivariateConditional(endog=[self.Italy_gdp], exog=[self.growth], dep_type='c', indep_type='c', bw='cv_ml')
        npt.assert_allclose(dens_ml.bw, [0.5341164, 0.04510836], atol=0.001)

    @pytest.mark.slow
    def test_unordered_CV_LS(self):
        dens_ls = nparam.KDEMultivariateConditional(endog=[self.oecd], exog=[self.growth], dep_type='u', indep_type='c', bw='cv_ls')

    def test_pdf_continuous(self):
        bw_cv_ml = np.array([0.010043, 12095254.7])
        dens = nparam.KDEMultivariateConditional(endog=[self.growth], exog=[self.Italy_gdp], dep_type='c', indep_type='c', bw=bw_cv_ml)
        sm_result = np.squeeze(dens.pdf()[0:5])
        R_result = [11.97964, 12.7329, 13.23037, 13.46438, 12.22779]
        npt.assert_allclose(sm_result, R_result, atol=0.001)

    @pytest.mark.slow
    def test_pdf_mixeddata(self):
        dens = nparam.KDEMultivariateConditional(endog=[self.Italy_gdp], exog=[self.Italy_year], dep_type='c', indep_type='o', bw='cv_ls')
        sm_result = np.squeeze(dens.pdf()[0:5])
        expected = [0.08592089, 0.0193275, 0.05310327, 0.09642667, 0.171954]
        npt.assert_allclose(sm_result, expected, atol=0, rtol=1e-05)

    def test_continuous_normal_ref(self):
        dens_nm = nparam.KDEMultivariateConditional(endog=[self.Italy_gdp], exog=[self.growth], dep_type='c', indep_type='c', bw='normal_reference')
        sm_result = dens_nm.bw
        R_result = [1.283532, 0.01535401]
        npt.assert_allclose(sm_result, R_result, atol=0.1)
        dens_nm2 = nparam.KDEMultivariateConditional(endog=[self.Italy_gdp], exog=[self.growth], dep_type='c', indep_type='c', bw=None)
        assert_allclose(dens_nm2.bw, dens_nm.bw, rtol=1e-10)
        assert_equal(dens_nm2._bw_method, 'normal_reference')
        repr(dens_nm2)

    def test_continuous_cdf(self):
        dens_nm = nparam.KDEMultivariateConditional(endog=[self.Italy_gdp], exog=[self.growth], dep_type='c', indep_type='c', bw='normal_reference')
        sm_result = dens_nm.cdf()[0:5]
        R_result = [0.8130492, 0.95046942, 0.86878727, 0.71961748, 0.38685423]
        npt.assert_allclose(sm_result, R_result, atol=0.001)

    @pytest.mark.slow
    def test_mixeddata_cdf(self):
        dens = nparam.KDEMultivariateConditional(endog=[self.Italy_gdp], exog=[self.Italy_year], dep_type='c', indep_type='o', bw='cv_ls')
        sm_result = dens.cdf()[0:5]
        expected = [0.83378885, 0.97684477, 0.90655143, 0.79393161, 0.43629083]
        npt.assert_allclose(sm_result, expected, atol=0, rtol=1e-05)

    @pytest.mark.slow
    def test_continuous_cvml_efficient(self):
        nobs = 500
        np.random.seed(12345)
        ovals = np.random.binomial(2, 0.5, size=(nobs,))
        C1 = np.random.normal(size=(nobs,))
        noise = np.random.normal(size=(nobs,))
        b0 = 3
        b1 = 1.2
        b2 = 3.7
        Y = b0 + b1 * C1 + b2 * ovals + noise
        dens_efficient = nparam.KDEMultivariateConditional(endog=[Y], exog=[C1], dep_type='c', indep_type='c', bw='cv_ml', defaults=nparam.EstimatorSettings(efficient=True, n_sub=50))
        bw_expected = np.array([0.73387, 0.43715])
        npt.assert_allclose(dens_efficient.bw, bw_expected, atol=0, rtol=0.001)

    def test_efficient_user_specified_bw(self):
        nobs = 400
        np.random.seed(12345)
        C1 = np.random.normal(size=(nobs,))
        C2 = np.random.normal(2, 1, size=(nobs,))
        bw_user = [0.23, 434697.22]
        dens = nparam.KDEMultivariate(data=[C1, C2], var_type='cc', bw=bw_user, defaults=nparam.EstimatorSettings(efficient=True, randomize=False, n_sub=100))
        npt.assert_equal(dens.bw, bw_user)