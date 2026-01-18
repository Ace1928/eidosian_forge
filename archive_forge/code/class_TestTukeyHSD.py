from itertools import product
import numpy as np
import random
import functools
import pytest
from numpy.testing import (assert_, assert_equal, assert_allclose,
from pytest import raises as assert_raises
import scipy.stats as stats
from scipy.stats import distributions
from scipy.stats._hypotests import (epps_singleton_2samp, cramervonmises,
from scipy.stats._mannwhitneyu import mannwhitneyu, _mwu_state
from .common_tests import check_named_results
from scipy._lib._testutils import _TestPythranFunc
class TestTukeyHSD:
    data_same_size = ([24.5, 23.5, 26.4, 27.1, 29.9], [28.4, 34.2, 29.5, 32.2, 30.1], [26.1, 28.3, 24.3, 26.2, 27.8])
    data_diff_size = ([24.5, 23.5, 26.28, 26.4, 27.1, 29.9, 30.1, 30.1], [28.4, 34.2, 29.5, 32.2, 30.1], [26.1, 28.3, 24.3, 26.2, 27.8])
    extreme_size = ([24.5, 23.5, 26.4], [28.4, 34.2, 29.5, 32.2, 30.1, 28.4, 34.2, 29.5, 32.2, 30.1], [26.1, 28.3, 24.3, 26.2, 27.8])
    sas_same_size = '\n    Comparison LowerCL Difference UpperCL Significance\n    2 - 3\t0.6908830568\t4.34\t7.989116943\t    1\n    2 - 1\t0.9508830568\t4.6 \t8.249116943 \t1\n    3 - 2\t-7.989116943\t-4.34\t-0.6908830568\t1\n    3 - 1\t-3.389116943\t0.26\t3.909116943\t    0\n    1 - 2\t-8.249116943\t-4.6\t-0.9508830568\t1\n    1 - 3\t-3.909116943\t-0.26\t3.389116943\t    0\n    '
    sas_diff_size = '\n    Comparison LowerCL Difference UpperCL Significance\n    2 - 1\t0.2679292645\t3.645\t7.022070736\t    1\n    2 - 3\t0.5934764007\t4.34\t8.086523599\t    1\n    1 - 2\t-7.022070736\t-3.645\t-0.2679292645\t1\n    1 - 3\t-2.682070736\t0.695\t4.072070736\t    0\n    3 - 2\t-8.086523599\t-4.34\t-0.5934764007\t1\n    3 - 1\t-4.072070736\t-0.695\t2.682070736\t    0\n    '
    sas_extreme = '\n    Comparison LowerCL Difference UpperCL Significance\n    2 - 3\t1.561605075\t    4.34\t7.118394925\t    1\n    2 - 1\t2.740784879\t    6.08\t9.419215121\t    1\n    3 - 2\t-7.118394925\t-4.34\t-1.561605075\t1\n    3 - 1\t-1.964526566\t1.74\t5.444526566\t    0\n    1 - 2\t-9.419215121\t-6.08\t-2.740784879\t1\n    1 - 3\t-5.444526566\t-1.74\t1.964526566\t    0\n    '

    @pytest.mark.parametrize('data,res_expect_str,atol', ((data_same_size, sas_same_size, 0.0001), (data_diff_size, sas_diff_size, 0.0001), (extreme_size, sas_extreme, 1e-10)), ids=['equal size sample', 'unequal sample size', 'extreme sample size differences'])
    def test_compare_sas(self, data, res_expect_str, atol):
        """
        SAS code used to generate results for each sample:
        DATA ACHE;
        INPUT BRAND RELIEF;
        CARDS;
        1 24.5
        ...
        3 27.8
        ;
        ods graphics on;   ODS RTF;ODS LISTING CLOSE;
           PROC ANOVA DATA=ACHE;
           CLASS BRAND;
           MODEL RELIEF=BRAND;
           MEANS BRAND/TUKEY CLDIFF;
           TITLE 'COMPARE RELIEF ACROSS MEDICINES  - ANOVA EXAMPLE';
           ods output  CLDiffs =tc;
        proc print data=tc;
            format LowerCL 17.16 UpperCL 17.16 Difference 17.16;
            title "Output with many digits";
        RUN;
        QUIT;
        ODS RTF close;
        ODS LISTING;
        """
        res_expect = np.asarray(res_expect_str.replace(' - ', ' ').split()[5:], dtype=float).reshape((6, 6))
        res_tukey = stats.tukey_hsd(*data)
        conf = res_tukey.confidence_interval()
        for i, j, l, s, h, sig in res_expect:
            i, j = (int(i) - 1, int(j) - 1)
            assert_allclose(conf.low[i, j], l, atol=atol)
            assert_allclose(res_tukey.statistic[i, j], s, atol=atol)
            assert_allclose(conf.high[i, j], h, atol=atol)
            assert_allclose(res_tukey.pvalue[i, j] <= 0.05, sig == 1)
    matlab_sm_siz = '\n        1\t2\t-8.2491590248597\t-4.6\t-0.9508409751403\t0.0144483269098\n        1\t3\t-3.9091590248597\t-0.26\t3.3891590248597\t0.9803107240900\n        2\t3\t0.6908409751403\t4.34\t7.9891590248597\t0.0203311368795\n        '
    matlab_diff_sz = '\n        1\t2\t-7.02207069748501\t-3.645\t-0.26792930251500 0.03371498443080\n        1\t3\t-2.68207069748500\t0.695\t4.07207069748500 0.85572267328807\n        2\t3\t0.59347644287720\t4.34\t8.08652355712281 0.02259047020620\n        '

    @pytest.mark.parametrize('data,res_expect_str,atol', ((data_same_size, matlab_sm_siz, 1e-12), (data_diff_size, matlab_diff_sz, 1e-07)), ids=['equal size sample', 'unequal size sample'])
    def test_compare_matlab(self, data, res_expect_str, atol):
        """
        vals = [24.5, 23.5,  26.4, 27.1, 29.9, 28.4, 34.2, 29.5, 32.2, 30.1,
         26.1, 28.3, 24.3, 26.2, 27.8]
        names = {'zero', 'zero', 'zero', 'zero', 'zero', 'one', 'one', 'one',
         'one', 'one', 'two', 'two', 'two', 'two', 'two'}
        [p,t,stats] = anova1(vals,names,"off");
        [c,m,h,nms] = multcompare(stats, "CType","hsd");
        """
        res_expect = np.asarray(res_expect_str.split(), dtype=float).reshape((3, 6))
        res_tukey = stats.tukey_hsd(*data)
        conf = res_tukey.confidence_interval()
        for i, j, l, s, h, p in res_expect:
            i, j = (int(i) - 1, int(j) - 1)
            assert_allclose(conf.low[i, j], l, atol=atol)
            assert_allclose(res_tukey.statistic[i, j], s, atol=atol)
            assert_allclose(conf.high[i, j], h, atol=atol)
            assert_allclose(res_tukey.pvalue[i, j], p, atol=atol)

    def test_compare_r(self):
        """
        Testing against results and p-values from R:
        from: https://www.rdocumentation.org/packages/stats/versions/3.6.2/
        topics/TukeyHSD
        > require(graphics)
        > summary(fm1 <- aov(breaks ~ tension, data = warpbreaks))
        > TukeyHSD(fm1, "tension", ordered = TRUE)
        > plot(TukeyHSD(fm1, "tension"))
        Tukey multiple comparisons of means
        95% family-wise confidence level
        factor levels have been ordered
        Fit: aov(formula = breaks ~ tension, data = warpbreaks)
        $tension
        """
        str_res = '\n                diff        lwr      upr     p adj\n        2 - 3  4.722222 -4.8376022 14.28205 0.4630831\n        1 - 3 14.722222  5.1623978 24.28205 0.0014315\n        1 - 2 10.000000  0.4401756 19.55982 0.0384598\n        '
        res_expect = np.asarray(str_res.replace(' - ', ' ').split()[5:], dtype=float).reshape((3, 6))
        data = ([26, 30, 54, 25, 70, 52, 51, 26, 67, 27, 14, 29, 19, 29, 31, 41, 20, 44], [18, 21, 29, 17, 12, 18, 35, 30, 36, 42, 26, 19, 16, 39, 28, 21, 39, 29], [36, 21, 24, 18, 10, 43, 28, 15, 26, 20, 21, 24, 17, 13, 15, 15, 16, 28])
        res_tukey = stats.tukey_hsd(*data)
        conf = res_tukey.confidence_interval()
        for i, j, s, l, h, p in res_expect:
            i, j = (int(i) - 1, int(j) - 1)
            assert_allclose(conf.low[i, j], l, atol=1e-07)
            assert_allclose(res_tukey.statistic[i, j], s, atol=1e-06)
            assert_allclose(conf.high[i, j], h, atol=1e-05)
            assert_allclose(res_tukey.pvalue[i, j], p, atol=1e-07)

    def test_engineering_stat_handbook(self):
        """
        Example sourced from:
        https://www.itl.nist.gov/div898/handbook/prc/section4/prc471.htm
        """
        group1 = [6.9, 5.4, 5.8, 4.6, 4.0]
        group2 = [8.3, 6.8, 7.8, 9.2, 6.5]
        group3 = [8.0, 10.5, 8.1, 6.9, 9.3]
        group4 = [5.8, 3.8, 6.1, 5.6, 6.2]
        res = stats.tukey_hsd(group1, group2, group3, group4)
        conf = res.confidence_interval()
        lower = np.asarray([[0, 0, 0, -2.25], [0.29, 0, -2.93, 0.13], [1.13, 0, 0, 0.97], [0, 0, 0, 0]])
        upper = np.asarray([[0, 0, 0, 1.93], [4.47, 0, 1.25, 4.31], [5.31, 0, 0, 5.15], [0, 0, 0, 0]])
        for i, j in [(1, 0), (2, 0), (0, 3), (1, 2), (2, 3)]:
            assert_allclose(conf.low[i, j], lower[i, j], atol=0.01)
            assert_allclose(conf.high[i, j], upper[i, j], atol=0.01)

    def test_rand_symm(self):
        np.random.seed(1234)
        data = np.random.rand(3, 100)
        res = stats.tukey_hsd(*data)
        conf = res.confidence_interval()
        assert_equal(conf.low, -conf.high.T)
        assert_equal(np.diagonal(conf.high), conf.high[0, 0])
        assert_equal(np.diagonal(conf.low), conf.low[0, 0])
        assert_equal(res.statistic, -res.statistic.T)
        assert_equal(np.diagonal(res.statistic), 0)
        assert_equal(res.pvalue, res.pvalue.T)
        assert_equal(np.diagonal(res.pvalue), 1)

    def test_no_inf(self):
        with assert_raises(ValueError, match='...must be finite.'):
            stats.tukey_hsd([1, 2, 3], [2, np.inf], [6, 7, 3])

    def test_is_1d(self):
        with assert_raises(ValueError, match='...must be one-dimensional'):
            stats.tukey_hsd([[1, 2], [2, 3]], [2, 5], [5, 23, 6])

    def test_no_empty(self):
        with assert_raises(ValueError, match='...must be greater than one'):
            stats.tukey_hsd([], [2, 5], [4, 5, 6])

    @pytest.mark.parametrize('nargs', (0, 1))
    def test_not_enough_treatments(self, nargs):
        with assert_raises(ValueError, match='...more than 1 treatment.'):
            stats.tukey_hsd(*[[23, 7, 3]] * nargs)

    @pytest.mark.parametrize('cl', [-0.5, 0, 1, 2])
    def test_conf_level_invalid(self, cl):
        with assert_raises(ValueError, match='must be between 0 and 1'):
            r = stats.tukey_hsd([23, 7, 3], [3, 4], [9, 4])
            r.confidence_interval(cl)

    def test_2_args_ttest(self):
        res_tukey = stats.tukey_hsd(*self.data_diff_size[:2])
        res_ttest = stats.ttest_ind(*self.data_diff_size[:2])
        assert_allclose(res_ttest.pvalue, res_tukey.pvalue[0, 1])
        assert_allclose(res_ttest.pvalue, res_tukey.pvalue[1, 0])