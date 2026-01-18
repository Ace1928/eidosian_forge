import pytest
import numpy as np
import numpy.testing as npt
import statsmodels.api as sm
class TestKernelReg(KernelRegressionTestBase):

    def test_ordered_lc_cvls(self):
        model = nparam.KernelReg(endog=[self.Italy_gdp], exog=[self.Italy_year], reg_type='lc', var_type='o', bw='cv_ls')
        sm_bw = model.bw
        R_bw = 0.1390096
        sm_mean, sm_mfx = model.fit()
        sm_mean = sm_mean[0:5]
        sm_mfx = sm_mfx[0:5]
        R_mean = 6.190486
        sm_R2 = model.r_squared()
        R_R2 = 0.1435323
        npt.assert_allclose(sm_bw, R_bw, atol=0.01)
        npt.assert_allclose(sm_mean, R_mean, atol=0.01)
        npt.assert_allclose(sm_R2, R_R2, atol=0.01)

    def test_continuousdata_lc_cvls(self):
        model = nparam.KernelReg(endog=[self.y], exog=[self.c1, self.c2], reg_type='lc', var_type='cc', bw='cv_ls')
        sm_bw = model.bw
        R_bw = [0.6163835, 0.1649656]
        sm_mean, sm_mfx = model.fit()
        sm_mean = sm_mean[0:5]
        sm_mfx = sm_mfx[0:5]
        R_mean = [31.49157, 37.29536, 43.72332, 40.58997, 36.80711]
        sm_R2 = model.r_squared()
        R_R2 = 0.956381720885
        npt.assert_allclose(sm_bw, R_bw, atol=0.01)
        npt.assert_allclose(sm_mean, R_mean, atol=0.01)
        npt.assert_allclose(sm_R2, R_R2, atol=0.01)

    def test_continuousdata_ll_cvls(self):
        model = nparam.KernelReg(endog=[self.y], exog=[self.c1, self.c2], reg_type='ll', var_type='cc', bw='cv_ls')
        sm_bw = model.bw
        R_bw = [1.717891, 2.449415]
        sm_mean, sm_mfx = model.fit()
        sm_mean = sm_mean[0:5]
        sm_mfx = sm_mfx[0:5]
        R_mean = [31.16003, 37.30323, 44.4987, 40.73704, 36.19083]
        sm_R2 = model.r_squared()
        R_R2 = 0.9336019
        npt.assert_allclose(sm_bw, R_bw, atol=0.01)
        npt.assert_allclose(sm_mean, R_mean, atol=0.01)
        npt.assert_allclose(sm_R2, R_R2, atol=0.01)

    def test_continuous_mfx_ll_cvls(self, file_name='RegData.csv'):
        nobs = 200
        np.random.seed(1234)
        C1 = np.random.normal(size=(nobs,))
        C2 = np.random.normal(2, 1, size=(nobs,))
        C3 = np.random.beta(0.5, 0.2, size=(nobs,))
        noise = np.random.normal(size=(nobs,))
        b0 = 3
        b1 = 1.2
        b2 = 3.7
        b3 = 2.3
        Y = b0 + b1 * C1 + b2 * C2 + b3 * C3 + noise
        bw_cv_ls = np.array([0.96075, 0.5682, 0.29835])
        model = nparam.KernelReg(endog=[Y], exog=[C1, C2, C3], reg_type='ll', var_type='ccc', bw=bw_cv_ls)
        sm_mean, sm_mfx = model.fit()
        sm_mean = sm_mean[0:5]
        npt.assert_allclose(sm_mfx[0, :], [b1, b2, b3], rtol=0.2)

    def test_mixed_mfx_ll_cvls(self, file_name='RegData.csv'):
        nobs = 200
        np.random.seed(1234)
        ovals = np.random.binomial(2, 0.5, size=(nobs,))
        C1 = np.random.normal(size=(nobs,))
        C2 = np.random.normal(2, 1, size=(nobs,))
        noise = np.random.normal(size=(nobs,))
        b0 = 3
        b1 = 1.2
        b2 = 3.7
        b3 = 2.3
        Y = b0 + b1 * C1 + b2 * C2 + b3 * ovals + noise
        bw_cv_ls = np.array([1.04726, 1.67485, 0.39852])
        model = nparam.KernelReg(endog=[Y], exog=[C1, C2, ovals], reg_type='ll', var_type='cco', bw=bw_cv_ls)
        sm_mean, sm_mfx = model.fit()
        sm_R2 = model.r_squared()
        npt.assert_allclose(sm_mfx[0, :], [b1, b2, b3], rtol=0.2)

    @pytest.mark.slow
    @pytest.mark.xfail(reason='Test does not make much sense - always passes with very small bw.')
    def test_mfx_nonlinear_ll_cvls(self, file_name='RegData.csv'):
        nobs = 200
        np.random.seed(1234)
        C1 = np.random.normal(size=(nobs,))
        C2 = np.random.normal(2, 1, size=(nobs,))
        C3 = np.random.beta(0.5, 0.2, size=(nobs,))
        noise = np.random.normal(size=(nobs,))
        b0 = 3
        b1 = 1.2
        b3 = 2.3
        Y = b0 + b1 * C1 * C2 + b3 * C3 + noise
        model = nparam.KernelReg(endog=[Y], exog=[C1, C2, C3], reg_type='ll', var_type='ccc', bw='cv_ls')
        sm_bw = model.bw
        sm_mean, sm_mfx = model.fit()
        sm_R2 = model.r_squared()
        mfx1 = b1 * C2
        mfx2 = b1 * C1
        npt.assert_allclose(sm_mean, Y, rtol=0.2)
        npt.assert_allclose(sm_mfx[:, 0], mfx1, rtol=0.2)
        npt.assert_allclose(sm_mfx[0:10, 1], mfx2[0:10], rtol=0.2)

    @pytest.mark.slow
    def test_continuous_cvls_efficient(self):
        nobs = 500
        np.random.seed(12345)
        C1 = np.random.normal(size=(nobs,))
        C2 = np.random.normal(2, 1, size=(nobs,))
        b0 = 3
        b1 = 1.2
        b2 = 3.7
        Y = b0 + b1 * C1 + b2 * C2
        model_efficient = nparam.KernelReg(endog=[Y], exog=[C1], reg_type='lc', var_type='c', bw='cv_ls', defaults=nparam.EstimatorSettings(efficient=True, n_sub=100))
        model = nparam.KernelReg(endog=[Y], exog=[C1], reg_type='ll', var_type='c', bw='cv_ls')
        npt.assert_allclose(model.bw, model_efficient.bw, atol=0.05, rtol=0.1)

    @pytest.mark.slow
    def test_censored_ll_cvls(self):
        nobs = 200
        np.random.seed(1234)
        C1 = np.random.normal(size=(nobs,))
        C2 = np.random.normal(2, 1, size=(nobs,))
        noise = np.random.normal(size=(nobs,))
        Y = 0.3 + 1.2 * C1 - 0.9 * C2 + noise
        Y[Y > 0] = 0
        model = nparam.KernelCensoredReg(endog=[Y], exog=[C1, C2], reg_type='ll', var_type='cc', bw='cv_ls', censor_val=0)
        sm_mean, sm_mfx = model.fit()
        npt.assert_allclose(sm_mfx[0, :], [1.2, -0.9], rtol=0.2)

    @pytest.mark.slow
    def test_continuous_lc_aic(self):
        nobs = 200
        np.random.seed(1234)
        C1 = np.random.normal(size=(nobs,))
        C2 = np.random.normal(2, 1, size=(nobs,))
        noise = np.random.normal(size=(nobs,))
        Y = 0.3 + 1.2 * C1 - 0.9 * C2 + noise
        model = nparam.KernelReg(endog=[Y], exog=[C1, C2], reg_type='lc', var_type='cc', bw='aic')
        bw_expected = [0.3987821, 0.50933458]
        npt.assert_allclose(model.bw, bw_expected, rtol=0.001)

    @pytest.mark.slow
    def test_significance_continuous(self):
        nobs = 250
        np.random.seed(12345)
        C1 = np.random.normal(size=(nobs,))
        C2 = np.random.normal(2, 1, size=(nobs,))
        C3 = np.random.beta(0.5, 0.2, size=(nobs,))
        noise = np.random.normal(size=(nobs,))
        b1 = 1.2
        b2 = 3.7
        Y = b1 * C1 + b2 * C2 + noise
        bw = [11108137.1087194, 1333821.85150218]
        model = nparam.KernelReg(endog=[Y], exog=[C1, C3], reg_type='ll', var_type='cc', bw=bw)
        nboot = 45
        sig_var12 = model.sig_test([0, 1], nboot=nboot)
        npt.assert_equal(sig_var12 == 'Not Significant', False)
        sig_var1 = model.sig_test([0], nboot=nboot)
        npt.assert_equal(sig_var1 == 'Not Significant', False)
        sig_var2 = model.sig_test([1], nboot=nboot)
        npt.assert_equal(sig_var2 == 'Not Significant', True)

    @pytest.mark.slow
    def test_significance_discrete(self):
        nobs = 200
        np.random.seed(12345)
        ovals = np.random.binomial(2, 0.5, size=(nobs,))
        C2 = np.random.normal(2, 1, size=(nobs,))
        C3 = np.random.beta(0.5, 0.2, size=(nobs,))
        noise = np.random.normal(size=(nobs,))
        b1 = 1.2
        b2 = 3.7
        Y = b1 * ovals + b2 * C2 + noise
        bw = [3.63473198, 1214048.03]
        model = nparam.KernelReg(endog=[Y], exog=[ovals, C3], reg_type='ll', var_type='oc', bw=bw)
        nboot = 45
        sig_var1 = model.sig_test([0], nboot=nboot)
        npt.assert_equal(sig_var1 == 'Not Significant', False)
        sig_var2 = model.sig_test([1], nboot=nboot)
        npt.assert_equal(sig_var2 == 'Not Significant', True)

    def test_user_specified_kernel(self):
        model = nparam.KernelReg(endog=[self.y], exog=[self.c1, self.c2], reg_type='ll', var_type='cc', bw='cv_ls', ckertype='tricube')
        sm_bw = model.bw
        R_bw = [0.581663, 0.5652]
        sm_mean, sm_mfx = model.fit()
        sm_mean = sm_mean[0:5]
        sm_mfx = sm_mfx[0:5]
        R_mean = [30.926714, 36.994604, 44.438358, 40.680598, 35.961593]
        sm_R2 = model.r_squared()
        R_R2 = 0.934825
        npt.assert_allclose(sm_bw, R_bw, atol=0.01)
        npt.assert_allclose(sm_mean, R_mean, atol=0.01)
        npt.assert_allclose(sm_R2, R_R2, atol=0.01)

    def test_censored_user_specified_kernel(self):
        model = nparam.KernelCensoredReg(endog=[self.y], exog=[self.c1, self.c2], reg_type='ll', var_type='cc', bw='cv_ls', censor_val=0, ckertype='tricube')
        sm_bw = model.bw
        R_bw = [0.581663, 0.5652]
        sm_mean, sm_mfx = model.fit()
        sm_mean = sm_mean[0:5]
        sm_mfx = sm_mfx[0:5]
        R_mean = [29.205526, 29.538008, 31.667581, 31.978866, 30.926714]
        sm_R2 = model.r_squared()
        R_R2 = 0.934825
        npt.assert_allclose(sm_bw, R_bw, atol=0.01)
        npt.assert_allclose(sm_mean, R_mean, atol=0.01)
        npt.assert_allclose(sm_R2, R_R2, atol=0.01)

    def test_efficient_user_specificed_bw(self):
        bw_user = [0.23, 434697.22]
        model = nparam.KernelReg(endog=[self.y], exog=[self.c1, self.c2], reg_type='lc', var_type='cc', bw=bw_user, defaults=nparam.EstimatorSettings(efficient=True))
        npt.assert_equal(model.bw, bw_user)

    def test_censored_efficient_user_specificed_bw(self):
        nobs = 200
        np.random.seed(1234)
        C1 = np.random.normal(size=(nobs,))
        C2 = np.random.normal(2, 1, size=(nobs,))
        noise = np.random.normal(size=(nobs,))
        Y = 0.3 + 1.2 * C1 - 0.9 * C2 + noise
        Y[Y > 0] = 0
        bw_user = [0.23, 434697.22]
        model = nparam.KernelCensoredReg(endog=[Y], exog=[C1, C2], reg_type='ll', var_type='cc', bw=bw_user, censor_val=0, defaults=nparam.EstimatorSettings(efficient=True))
        npt.assert_equal(model.bw, bw_user)