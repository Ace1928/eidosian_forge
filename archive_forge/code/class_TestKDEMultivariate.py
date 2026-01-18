import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
class TestKDEMultivariate(KDETestBase):

    @pytest.mark.slow
    def test_pdf_mixeddata_CV_LS(self):
        dens_u = nparam.KDEMultivariate(data=[self.c1, self.o, self.o2], var_type='coo', bw='cv_ls')
        npt.assert_allclose(dens_u.bw, [0.70949447, 0.08736727, 0.09220476], atol=1e-06)

    @pytest.mark.slow
    def test_pdf_mixeddata_LS_vs_ML(self):
        dens_ls = nparam.KDEMultivariate(data=[self.c1, self.o, self.o2], var_type='coo', bw='cv_ls')
        dens_ml = nparam.KDEMultivariate(data=[self.c1, self.o, self.o2], var_type='coo', bw='cv_ml')
        npt.assert_allclose(dens_ls.bw, dens_ml.bw, atol=0, rtol=0.5)

    def test_pdf_mixeddata_CV_ML(self):
        dens_ml = nparam.KDEMultivariate(data=[self.c1, self.o, self.c2], var_type='coc', bw='cv_ml')
        R_bw = [1.021563, 2.806409e-14, 0.5142077]
        npt.assert_allclose(dens_ml.bw, R_bw, atol=0.1, rtol=0.1)

    @pytest.mark.slow
    def test_pdf_continuous(self):
        dens = nparam.KDEMultivariate(data=[self.growth, self.Italy_gdp], var_type='cc', bw='cv_ls')
        sm_result = np.squeeze(dens.pdf()[0:5])
        R_result = [1.6202284, 0.7914245, 1.6084174, 2.4987204, 1.3705258]
        npt.assert_allclose(sm_result, R_result, atol=0.001)

    def test_pdf_ordered(self):
        dens = nparam.KDEMultivariate(data=[self.oecd], var_type='o', bw='cv_ls')
        sm_result = np.squeeze(dens.pdf()[0:5])
        R_result = [0.7236395, 0.7236395, 0.2763605, 0.2763605, 0.7236395]
        npt.assert_allclose(sm_result, R_result, atol=0.1)

    @pytest.mark.slow
    def test_unordered_CV_LS(self):
        dens = nparam.KDEMultivariate(data=[self.growth, self.oecd], var_type='cu', bw='cv_ls')
        R_result = [0.0052051, 0.05835941]
        npt.assert_allclose(dens.bw, R_result, atol=0.01)

    def test_continuous_cdf(self, data_predict=None):
        dens = nparam.KDEMultivariate(data=[self.Italy_gdp, self.growth], var_type='cc', bw='cv_ml')
        sm_result = dens.cdf()[0:5]
        R_result = [0.19218077, 0.299505196, 0.557303666, 0.513387712, 0.21098535]
        npt.assert_allclose(sm_result, R_result, atol=0.001)

    def test_mixeddata_cdf(self, data_predict=None):
        dens = nparam.KDEMultivariate(data=[self.Italy_gdp, self.oecd], var_type='cu', bw='cv_ml')
        sm_result = dens.cdf()[0:5]
        R_result = [0.5470001, 0.65907039, 0.89676865, 0.74132941, 0.25291361]
        npt.assert_allclose(sm_result, R_result, atol=0.001)

    @pytest.mark.slow
    def test_continuous_cvls_efficient(self):
        nobs = 400
        np.random.seed(12345)
        C1 = np.random.normal(size=(nobs,))
        C2 = np.random.normal(2, 1, size=(nobs,))
        Y = 0.3 + 1.2 * C1 - 0.9 * C2
        dens_efficient = nparam.KDEMultivariate(data=[Y, C1], var_type='cc', bw='cv_ls', defaults=nparam.EstimatorSettings(efficient=True, n_sub=100))
        bw = np.array([0.3404, 0.1666])
        npt.assert_allclose(bw, dens_efficient.bw, atol=0.1, rtol=0.2)

    @pytest.mark.slow
    def test_continuous_cvml_efficient(self):
        nobs = 400
        np.random.seed(12345)
        C1 = np.random.normal(size=(nobs,))
        C2 = np.random.normal(2, 1, size=(nobs,))
        Y = 0.3 + 1.2 * C1 - 0.9 * C2
        dens_efficient = nparam.KDEMultivariate(data=[Y, C1], var_type='cc', bw='cv_ml', defaults=nparam.EstimatorSettings(efficient=True, n_sub=100))
        bw = np.array([0.4471, 0.2861])
        npt.assert_allclose(bw, dens_efficient.bw, atol=0.1, rtol=0.2)

    @pytest.mark.slow
    def test_efficient_notrandom(self):
        nobs = 400
        np.random.seed(12345)
        C1 = np.random.normal(size=(nobs,))
        C2 = np.random.normal(2, 1, size=(nobs,))
        Y = 0.3 + 1.2 * C1 - 0.9 * C2
        dens_efficient = nparam.KDEMultivariate(data=[Y, C1], var_type='cc', bw='cv_ml', defaults=nparam.EstimatorSettings(efficient=True, randomize=False, n_sub=100))
        dens = nparam.KDEMultivariate(data=[Y, C1], var_type='cc', bw='cv_ml')
        npt.assert_allclose(dens.bw, dens_efficient.bw, atol=0.1, rtol=0.2)

    def test_efficient_user_specified_bw(self):
        nobs = 400
        np.random.seed(12345)
        C1 = np.random.normal(size=(nobs,))
        C2 = np.random.normal(2, 1, size=(nobs,))
        bw_user = [0.23, 434697.22]
        dens = nparam.KDEMultivariate(data=[C1, C2], var_type='cc', bw=bw_user, defaults=nparam.EstimatorSettings(efficient=True, randomize=False, n_sub=100))
        npt.assert_equal(dens.bw, bw_user)