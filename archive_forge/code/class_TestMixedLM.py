from statsmodels.compat.platform import PLATFORM_OSX
import os
import csv
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
import pytest
from statsmodels.regression.mixed_linear_model import (
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from statsmodels.base import _penalties as penalties
import statsmodels.tools.numdiff as nd
from .results import lme_r_results
class TestMixedLM:

    @pytest.mark.slow
    @pytest.mark.parametrize('use_sqrt', [False, True])
    @pytest.mark.parametrize('reml', [False, True])
    @pytest.mark.parametrize('profile_fe', [False, True])
    def test_compare_numdiff(self, use_sqrt, reml, profile_fe):
        n_grp = 200
        grpsize = 5
        k_fe = 3
        k_re = 2
        np.random.seed(3558)
        exog_fe = np.random.normal(size=(n_grp * grpsize, k_fe))
        exog_re = np.random.normal(size=(n_grp * grpsize, k_re))
        exog_re[:, 0] = 1
        exog_vc = np.random.normal(size=(n_grp * grpsize, 3))
        slopes = np.random.normal(size=(n_grp, k_re))
        slopes[:, -1] *= 2
        slopes = np.kron(slopes, np.ones((grpsize, 1)))
        slopes_vc = np.random.normal(size=(n_grp, 3))
        slopes_vc = np.kron(slopes_vc, np.ones((grpsize, 1)))
        slopes_vc[:, -1] *= 2
        re_values = (slopes * exog_re).sum(1)
        vc_values = (slopes_vc * exog_vc).sum(1)
        err = np.random.normal(size=n_grp * grpsize)
        endog = exog_fe.sum(1) + re_values + vc_values + err
        groups = np.kron(range(n_grp), np.ones(grpsize))
        vc = {'a': {}, 'b': {}}
        for i in range(n_grp):
            ix = np.flatnonzero(groups == i)
            vc['a'][i] = exog_vc[ix, 0:2]
            vc['b'][i] = exog_vc[ix, 2:3]
        with pytest.warns(UserWarning, match='Using deprecated variance'):
            model = MixedLM(endog, exog_fe, groups, exog_re, exog_vc=vc, use_sqrt=use_sqrt)
        rslt = model.fit(reml=reml)
        loglike = loglike_function(model, profile_fe=profile_fe, has_fe=not profile_fe)
        try:
            for kr in range(5):
                fe_params = np.random.normal(size=k_fe)
                cov_re = np.random.normal(size=(k_re, k_re))
                cov_re = np.dot(cov_re.T, cov_re)
                vcomp = np.random.normal(size=2) ** 2
                params = MixedLMParams.from_components(fe_params, cov_re=cov_re, vcomp=vcomp)
                params_vec = params.get_packed(has_fe=not profile_fe, use_sqrt=use_sqrt)
                gr = -model.score(params, profile_fe=profile_fe)
                ngr = nd.approx_fprime(params_vec, loglike)
                assert_allclose(gr, ngr, rtol=0.001)
            if profile_fe is False and use_sqrt is False:
                hess, sing = model.hessian(rslt.params_object)
                if sing:
                    pytest.fail('hessian should not be singular')
                hess *= -1
                params_vec = rslt.params_object.get_packed(use_sqrt=False, has_fe=True)
                loglike_h = loglike_function(model, profile_fe=False, has_fe=True)
                nhess = nd.approx_hess(params_vec, loglike_h)
                assert_allclose(hess, nhess, rtol=0.001)
        except AssertionError:
            if PLATFORM_OSX:
                pytest.xfail('fails on OSX due to unresolved numerical differences')
            else:
                raise

    def test_default_re(self):
        np.random.seed(3235)
        exog = np.random.normal(size=(300, 4))
        groups = np.kron(np.arange(100), [1, 1, 1])
        g_errors = np.kron(np.random.normal(size=100), [1, 1, 1])
        endog = exog.sum(1) + g_errors + np.random.normal(size=300)
        mdf1 = MixedLM(endog, exog, groups).fit()
        mdf2 = MixedLM(endog, exog, groups, np.ones(300)).fit()
        assert_almost_equal(mdf1.params, mdf2.params, decimal=8)

    def test_history(self):
        np.random.seed(3235)
        exog = np.random.normal(size=(300, 4))
        groups = np.kron(np.arange(100), [1, 1, 1])
        g_errors = np.kron(np.random.normal(size=100), [1, 1, 1])
        endog = exog.sum(1) + g_errors + np.random.normal(size=300)
        mod = MixedLM(endog, exog, groups)
        rslt = mod.fit(full_output=True)
        assert_equal(hasattr(rslt, 'hist'), True)

    @pytest.mark.slow
    @pytest.mark.smoke
    def test_profile_inference(self):
        np.random.seed(9814)
        k_fe = 2
        gsize = 3
        n_grp = 100
        exog = np.random.normal(size=(n_grp * gsize, k_fe))
        exog_re = np.ones((n_grp * gsize, 1))
        groups = np.kron(np.arange(n_grp), np.ones(gsize))
        vca = np.random.normal(size=n_grp * gsize)
        vcb = np.random.normal(size=n_grp * gsize)
        errors = 0
        g_errors = np.kron(np.random.normal(size=100), np.ones(gsize))
        errors += g_errors + exog_re[:, 0]
        rc = np.random.normal(size=n_grp)
        errors += np.kron(rc, np.ones(gsize)) * vca
        rc = np.random.normal(size=n_grp)
        errors += np.kron(rc, np.ones(gsize)) * vcb
        errors += np.random.normal(size=n_grp * gsize)
        endog = exog.sum(1) + errors
        vc = {'a': {}, 'b': {}}
        for k in range(n_grp):
            ii = np.flatnonzero(groups == k)
            vc['a'][k] = vca[ii][:, None]
            vc['b'][k] = vcb[ii][:, None]
        with pytest.warns(UserWarning, match='Using deprecated variance'):
            rslt = MixedLM(endog, exog, groups=groups, exog_re=exog_re, exog_vc=vc).fit()
        rslt.profile_re(0, vtype='re', dist_low=1, num_low=3, dist_high=1, num_high=3)
        rslt.profile_re('b', vtype='vc', dist_low=0.5, num_low=3, dist_high=0.5, num_high=3)

    def test_vcomp_1(self):
        np.random.seed(4279)
        exog = np.random.normal(size=(400, 1))
        exog_re = np.random.normal(size=(400, 2))
        groups = np.kron(np.arange(100), np.ones(4))
        slopes = np.random.normal(size=(100, 2))
        slopes[:, 1] *= 2
        slopes = np.kron(slopes, np.ones((4, 1))) * exog_re
        errors = slopes.sum(1) + np.random.normal(size=400)
        endog = exog.sum(1) + errors
        free = MixedLMParams(1, 2, 0)
        free.fe_params = np.ones(1)
        free.cov_re = np.eye(2)
        free.vcomp = np.zeros(0)
        model1 = MixedLM(endog, exog, groups, exog_re=exog_re)
        result1 = model1.fit(free=free)
        exog_vc = {'a': {}, 'b': {}}
        for k, group in enumerate(model1.group_labels):
            ix = model1.row_indices[group]
            exog_vc['a'][group] = exog_re[ix, 0:1]
            exog_vc['b'][group] = exog_re[ix, 1:2]
        with pytest.warns(UserWarning, match='Using deprecated variance'):
            model2 = MixedLM(endog, exog, groups, exog_vc=exog_vc)
        result2 = model2.fit()
        result2.summary()
        assert_allclose(result1.fe_params, result2.fe_params, atol=0.0001)
        assert_allclose(np.diag(result1.cov_re), result2.vcomp, atol=0.01, rtol=0.0001)
        assert_allclose(result1.bse[[0, 1, 3]], result2.bse, atol=0.01, rtol=0.01)

    def test_vcomp_2(self):
        np.random.seed(6241)
        n = 1600
        exog = np.random.normal(size=(n, 2))
        groups = np.kron(np.arange(n / 16), np.ones(16))
        errors = 0
        exog_re = np.random.normal(size=(n, 2))
        slopes = np.random.normal(size=(n // 16, 2))
        slopes = np.kron(slopes, np.ones((16, 1))) * exog_re
        errors += slopes.sum(1)
        subgroups1 = np.kron(np.arange(n / 4), np.ones(4))
        errors += np.kron(2 * np.random.normal(size=n // 4), np.ones(4))
        subgroups2 = np.kron(np.arange(n / 2), np.ones(2))
        errors += np.kron(2 * np.random.normal(size=n // 2), np.ones(2))
        errors += np.random.normal(size=n)
        endog = exog.sum(1) + errors
        df = pd.DataFrame(index=range(n))
        df['y'] = endog
        df['groups'] = groups
        df['x1'] = exog[:, 0]
        df['x2'] = exog[:, 1]
        df['z1'] = exog_re[:, 0]
        df['z2'] = exog_re[:, 1]
        df['v1'] = subgroups1
        df['v2'] = subgroups2
        vcf = {'a': '0 + C(v1)', 'b': '0 + C(v2)'}
        model1 = MixedLM.from_formula('y ~ x1 + x2', groups=groups, re_formula='0+z1+z2', vc_formula=vcf, data=df)
        result1 = model1.fit()
        assert_allclose(result1.fe_params, [0.16527, 0.99911, 0.96217], rtol=0.0001)
        assert_allclose(result1.cov_re, [[1.244, 0.146], [0.146, 1.371]], rtol=0.001)
        assert_allclose(result1.vcomp, [4.024, 3.997], rtol=0.001)
        assert_allclose(result1.bse.iloc[0:3], [0.1261, 0.03938, 0.03848], rtol=0.001)

    def test_vcomp_3(self):
        np.random.seed(4279)
        x1 = np.random.normal(size=400)
        groups = np.kron(np.arange(100), np.ones(4))
        slopes = np.random.normal(size=100)
        slopes = np.kron(slopes, np.ones(4)) * x1
        y = slopes + np.random.normal(size=400)
        vc_fml = {'a': '0 + x1'}
        df = pd.DataFrame({'y': y, 'x1': x1, 'groups': groups})
        model = MixedLM.from_formula('y ~ 1', groups='groups', vc_formula=vc_fml, data=df)
        result = model.fit()
        result.summary()
        assert_allclose(result.resid.iloc[0:4], np.r_[-1.180753, 0.279966, 0.578576, -0.667916], rtol=0.001)
        assert_allclose(result.fittedvalues.iloc[0:4], np.r_[-0.101549, 0.028613, -0.224621, -0.126295], rtol=0.001)

    def test_sparse(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        rdir = os.path.join(cur_dir, 'results')
        fname = os.path.join(rdir, 'pastes.csv')
        data = pd.read_csv(fname)
        vcf = {'cask': '0 + cask'}
        model = MixedLM.from_formula('strength ~ 1', groups='batch', re_formula='1', vc_formula=vcf, data=data)
        result = model.fit()
        model2 = MixedLM.from_formula('strength ~ 1', groups='batch', re_formula='1', vc_formula=vcf, use_sparse=True, data=data)
        result2 = model2.fit()
        assert_allclose(result.params, result2.params)
        assert_allclose(result.bse, result2.bse)

    def test_dietox(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        rdir = os.path.join(cur_dir, 'results')
        fname = os.path.join(rdir, 'dietox.csv')
        data = pd.read_csv(fname)
        model = MixedLM.from_formula('Weight ~ Time', groups='Pig', data=data)
        result = model.fit()
        assert_allclose(result.fe_params, np.r_[15.723523, 6.942505], rtol=1e-05)
        assert_allclose(result.bse[0:2], np.r_[0.78805374, 0.03338727], rtol=1e-05)
        assert_allclose(result.scale, 11.36692, rtol=1e-05)
        assert_allclose(result.cov_re, 40.39395, rtol=1e-05)
        assert_allclose(model.loglike(result.params_object), -2404.775, rtol=1e-05)
        data = pd.read_csv(fname)
        model = MixedLM.from_formula('Weight ~ Time', groups='Pig', data=data)
        result = model.fit(reml=False)
        assert_allclose(result.fe_params, np.r_[15.723517, 6.942506], rtol=1e-05)
        assert_allclose(result.bse[0:2], np.r_[0.7829397, 0.0333661], rtol=1e-05)
        assert_allclose(result.scale, 11.35251, rtol=1e-05)
        assert_allclose(result.cov_re, 39.82097, rtol=1e-05)
        assert_allclose(model.loglike(result.params_object), -2402.932, rtol=1e-05)

    def test_dietox_slopes(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        rdir = os.path.join(cur_dir, 'results')
        fname = os.path.join(rdir, 'dietox.csv')
        data = pd.read_csv(fname)
        model = MixedLM.from_formula('Weight ~ Time', groups='Pig', re_formula='1 + Time', data=data)
        result = model.fit(method='cg')
        assert_allclose(result.fe_params, np.r_[15.73865, 6.939014], rtol=1e-05)
        assert_allclose(result.bse[0:2], np.r_[0.5501253, 0.0798254], rtol=0.001)
        assert_allclose(result.scale, 6.03745, rtol=0.001)
        assert_allclose(result.cov_re.values.ravel(), np.r_[19.4934552, 0.2938323, 0.2938323, 0.416062], rtol=0.1)
        assert_allclose(model.loglike(result.params_object), -2217.047, rtol=1e-05)
        data = pd.read_csv(fname)
        model = MixedLM.from_formula('Weight ~ Time', groups='Pig', re_formula='1 + Time', data=data)
        result = model.fit(method='cg', reml=False)
        assert_allclose(result.fe_params, np.r_[15.73863, 6.93902], rtol=1e-05)
        assert_allclose(result.bse[0:2], np.r_[0.54629282, 0.07926954], rtol=0.001)
        assert_allclose(result.scale, 6.037441, rtol=0.001)
        assert_allclose(result.cov_re.values.ravel(), np.r_[19.190922, 0.293568, 0.293568, 0.409695], rtol=0.01)
        assert_allclose(model.loglike(result.params_object), -2215.753, rtol=1e-05)

    def test_pastes_vcomp(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        rdir = os.path.join(cur_dir, 'results')
        fname = os.path.join(rdir, 'pastes.csv')
        data = pd.read_csv(fname)
        vcf = {'cask': '0 + cask'}
        model = MixedLM.from_formula('strength ~ 1', groups='batch', re_formula='1', vc_formula=vcf, data=data)
        result = model.fit()
        assert_allclose(result.fe_params.iloc[0], 60.0533, rtol=0.001)
        assert_allclose(result.bse.iloc[0], 0.6769, rtol=0.001)
        assert_allclose(result.cov_re.iloc[0, 0], 1.657, rtol=0.001)
        assert_allclose(result.scale, 0.678, rtol=0.001)
        assert_allclose(result.llf, -123.49, rtol=0.1)
        assert_equal(result.aic, np.nan)
        assert_equal(result.bic, np.nan)
        resid = np.r_[0.17133538, -0.02866462, -1.08662875, 1.11337125, -0.12093607]
        assert_allclose(result.resid[0:5], resid, rtol=0.001)
        fit = np.r_[62.62866, 62.62866, 61.18663, 61.18663, 62.82094]
        assert_allclose(result.fittedvalues[0:5], fit, rtol=0.0001)
        model = MixedLM.from_formula('strength ~ 1', groups='batch', re_formula='1', vc_formula=vcf, data=data)
        result = model.fit(reml=False)
        assert_allclose(result.fe_params.iloc[0], 60.0533, rtol=0.001)
        assert_allclose(result.bse.iloc[0], 0.642, rtol=0.001)
        assert_allclose(result.cov_re.iloc[0, 0], 1.199, rtol=0.001)
        assert_allclose(result.scale, 0.67799, rtol=0.001)
        assert_allclose(result.llf, -123.997, rtol=0.1)
        assert_allclose(result.aic, 255.9944, rtol=0.001)
        assert_allclose(result.bic, 264.3718, rtol=0.001)

    @pytest.mark.slow
    def test_vcomp_formula(self):
        np.random.seed(6241)
        n = 800
        exog = np.random.normal(size=(n, 2))
        exog[:, 0] = 1
        ex_vc = []
        groups = np.kron(np.arange(n / 4), np.ones(4))
        errors = 0
        exog_re = np.random.normal(size=(n, 2))
        slopes = np.random.normal(size=(n // 4, 2))
        slopes = np.kron(slopes, np.ones((4, 1))) * exog_re
        errors += slopes.sum(1)
        ex_vc = np.random.normal(size=(n, 4))
        slopes = np.random.normal(size=(n // 4, 4))
        slopes[:, 2:] *= 2
        slopes = np.kron(slopes, np.ones((4, 1))) * ex_vc
        errors += slopes.sum(1)
        errors += np.random.normal(size=n)
        endog = exog.sum(1) + errors
        exog_vc = {'a': {}, 'b': {}}
        for k, group in enumerate(range(int(n / 4))):
            ix = np.flatnonzero(groups == group)
            exog_vc['a'][group] = ex_vc[ix, 0:2]
            exog_vc['b'][group] = ex_vc[ix, 2:]
        with pytest.warns(UserWarning, match='Using deprecated variance'):
            model1 = MixedLM(endog, exog, groups, exog_re=exog_re, exog_vc=exog_vc)
        result1 = model1.fit()
        df = pd.DataFrame(exog[:, 1:], columns=['x1'])
        df['y'] = endog
        df['re1'] = exog_re[:, 0]
        df['re2'] = exog_re[:, 1]
        df['vc1'] = ex_vc[:, 0]
        df['vc2'] = ex_vc[:, 1]
        df['vc3'] = ex_vc[:, 2]
        df['vc4'] = ex_vc[:, 3]
        vc_formula = {'a': '0 + vc1 + vc2', 'b': '0 + vc3 + vc4'}
        model2 = MixedLM.from_formula('y ~ x1', groups=groups, re_formula='0 + re1 + re2', vc_formula=vc_formula, data=df)
        result2 = model2.fit()
        assert_allclose(result1.fe_params, result2.fe_params, rtol=1e-08)
        assert_allclose(result1.cov_re, result2.cov_re, rtol=1e-08)
        assert_allclose(result1.vcomp, result2.vcomp, rtol=1e-08)
        assert_allclose(result1.params, result2.params, rtol=1e-08)
        assert_allclose(result1.bse, result2.bse, rtol=1e-08)

    def test_formulas(self):
        np.random.seed(2410)
        exog = np.random.normal(size=(300, 4))
        exog_re = np.random.normal(size=300)
        groups = np.kron(np.arange(100), [1, 1, 1])
        g_errors = exog_re * np.kron(np.random.normal(size=100), [1, 1, 1])
        endog = exog.sum(1) + g_errors + np.random.normal(size=300)
        mod1 = MixedLM(endog, exog, groups, exog_re)
        assert_(mod1.data.xnames == ['x1', 'x2', 'x3', 'x4'])
        assert_(mod1.data.exog_re_names == ['x_re1'])
        assert_(mod1.data.exog_re_names_full == ['x_re1 Var'])
        rslt1 = mod1.fit()
        df = pd.DataFrame({'endog': endog})
        for k in range(exog.shape[1]):
            df['exog%d' % k] = exog[:, k]
        df['exog_re'] = exog_re
        fml = 'endog ~ 0 + exog0 + exog1 + exog2 + exog3'
        re_fml = '0 + exog_re'
        mod2 = MixedLM.from_formula(fml, df, re_formula=re_fml, groups=groups)
        assert_(mod2.data.xnames == ['exog0', 'exog1', 'exog2', 'exog3'])
        assert_(mod2.data.exog_re_names == ['exog_re'])
        assert_(mod2.data.exog_re_names_full == ['exog_re Var'])
        rslt2 = mod2.fit()
        assert_almost_equal(rslt1.params, rslt2.params)
        df['groups'] = groups
        mod3 = MixedLM.from_formula(fml, df, re_formula=re_fml, groups='groups')
        assert_(mod3.data.xnames == ['exog0', 'exog1', 'exog2', 'exog3'])
        assert_(mod3.data.exog_re_names == ['exog_re'])
        assert_(mod3.data.exog_re_names_full == ['exog_re Var'])
        rslt3 = mod3.fit(start_params=rslt2.params)
        assert_allclose(rslt1.params, rslt3.params, rtol=0.0001)
        exog_re = np.ones(len(endog), dtype=np.float64)
        mod4 = MixedLM(endog, exog, groups, exog_re)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rslt4 = mod4.fit()
        from statsmodels.formula.api import mixedlm
        mod5 = mixedlm(fml, df, groups='groups')
        assert_(mod5.data.exog_re_names == ['groups'])
        assert_(mod5.data.exog_re_names_full == ['groups Var'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rslt5 = mod5.fit()
        assert_almost_equal(rslt4.params, rslt5.params)

    @pytest.mark.slow
    def test_regularized(self):
        np.random.seed(3453)
        exog = np.random.normal(size=(400, 5))
        groups = np.kron(np.arange(100), np.ones(4))
        expected_endog = exog[:, 0] - exog[:, 2]
        endog = expected_endog + np.kron(np.random.normal(size=100), np.ones(4)) + np.random.normal(size=400)
        md = MixedLM(endog, exog, groups)
        mdf1 = md.fit_regularized(alpha=1.0)
        mdf1.summary()
        md = MixedLM(endog, exog, groups)
        mdf2 = md.fit_regularized(alpha=10 * np.ones(5))
        mdf2.summary()
        pen = penalties.L2()
        mdf3 = md.fit_regularized(method=pen, alpha=0.0)
        mdf3.summary()
        pen = penalties.L2()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mdf4 = md.fit_regularized(method=pen, alpha=10.0)
        mdf4.summary()
        pen = penalties.PseudoHuber(0.3)
        mdf5 = md.fit_regularized(method=pen, alpha=1.0)
        mdf5.summary()