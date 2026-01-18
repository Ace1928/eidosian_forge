import os
import re
import warnings
from collections import namedtuple
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib
from numpy.testing import (assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize
from .common_tests import check_named_results
from scipy.spatial.distance import cdist
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
from scipy._lib._util import AxisError
class TestAlexanderGovern:

    def test_compare_dtypes(self):
        args = [[13, 13, 13, 13, 13, 13, 13, 12, 12], [14, 13, 12, 12, 12, 12, 12, 11, 11], [14, 14, 13, 13, 13, 13, 13, 12, 12], [15, 14, 13, 13, 13, 12, 12, 12, 11]]
        args_int16 = np.array(args, dtype=np.int16)
        args_int32 = np.array(args, dtype=np.int32)
        args_uint8 = np.array(args, dtype=np.uint8)
        args_float64 = np.array(args, dtype=np.float64)
        res_int16 = stats.alexandergovern(*args_int16)
        res_int32 = stats.alexandergovern(*args_int32)
        res_unit8 = stats.alexandergovern(*args_uint8)
        res_float64 = stats.alexandergovern(*args_float64)
        assert res_int16.pvalue == res_int32.pvalue == res_unit8.pvalue == res_float64.pvalue
        assert res_int16.statistic == res_int32.statistic == res_unit8.statistic == res_float64.statistic

    def test_bad_inputs(self):
        with assert_raises(ValueError, match='Input sample size must be greater than one.'):
            stats.alexandergovern([1, 2], [])
        with assert_raises(ValueError, match='Input sample size must be greater than one.'):
            stats.alexandergovern([1, 2], 2)
        with assert_raises(ValueError, match='Input sample size must be greater than one.'):
            stats.alexandergovern([1, 2], [2])
        with assert_raises(ValueError, match='Input samples must be finite.'):
            stats.alexandergovern([1, 2], [np.inf, np.inf])
        with assert_raises(ValueError, match='Input samples must be one-dimensional'):
            stats.alexandergovern([1, 2], [[1, 2], [3, 4]])

    def test_compare_r(self):
        """
        Data generated in R with
        > set.seed(1)
        > library("onewaytests")
        > library("tibble")
        > y <- c(rnorm(40, sd=10),
        +        rnorm(30, sd=15),
        +        rnorm(20, sd=20))
        > x <- c(rep("one", times=40),
        +        rep("two", times=30),
        +        rep("eight", times=20))
        > x <- factor(x)
        > ag.test(y ~ x, tibble(y,x))

        Alexander-Govern Test (alpha = 0.05)
        -------------------------------------------------------------
        data : y and x

        statistic  : 1.359941
        parameter  : 2
        p.value    : 0.5066321

        Result     : Difference is not statistically significant.
        -------------------------------------------------------------
        Example adapted from:
        https://eval-serv2.metpsy.uni-jena.de/wiki-metheval-hp/index.php/R_FUN_Alexander-Govern

        """
        one = [-6.264538107423324, 1.8364332422208225, -8.356286124100471, 15.952808021377916, 3.295077718153605, -8.204683841180152, 4.874290524284853, 7.383247051292173, 5.757813516534923, -3.0538838715635603, 15.11781168450848, 3.898432364114311, -6.2124058054180376, -22.146998871774997, 11.249309181431082, -0.4493360901523085, -0.16190263098946087, 9.438362106852992, 8.212211950980885, 5.939013212175088, 9.189773716082183, 7.821363007310671, 0.745649833651906, -19.89351695863373, 6.198257478947102, -0.5612873952900078, -1.557955067053293, -14.707523838992744, -4.781500551086204, 4.179415601997024, 13.58679551529044, -1.0278772734299553, 3.876716115593691, -0.5380504058290512, -13.770595568286065, -4.149945632996798, -3.942899537103493, -0.5931339671118566, 11.000253719838831, 7.631757484575442]
        two = [-2.4678539438038034, -3.8004252020476135, 10.454450631071062, 8.34994798010486, -10.331335418242798, -10.612427354431794, 5.468729432052455, 11.527993867731237, -1.6851931822534207, 13.216615896813222, 5.971588205506021, -9.180395898761569, 5.116795371366372, -16.94044644121189, 21.495355525515556, 29.7059984775879, -5.508322146997636, -15.662019394747961, 8.545794411636193, -2.0258190582123654, 36.024266407571645, -0.5886000409975387, 10.346090436761651, 0.4200323817099909, -11.14909813323608, 2.8318844927151434, -27.074379433365568, 21.98332292344329, 2.2988000731784655, 32.58917505543229]
        eight = [9.510190577993251, -14.198928618436291, 12.214527069781099, -18.68195263288503, -25.07266800478204, 5.828924710349257, -8.86583746436866, 0.02210703263248262, 1.4868264830332811, -11.79041892376144, -11.37337465637004, -2.7035723024766414, 23.56173993146409, -30.47133600859524, 11.878923752568431, 6.659007424270365, 21.261996745527256, -6.083678472686013, 7.400376198325763, 5.341975815444621]
        soln = stats.alexandergovern(one, two, eight)
        assert_allclose(soln.statistic, 1.359940544799945)
        assert_allclose(soln.pvalue, 0.5066320530967644)

    def test_compare_scholar(self):
        """
        Data taken from 'The Modification and Evaluation of the
        Alexander-Govern Test in Terms of Power' by Kingsley Ochuko, T.,
        Abdullah, S., Binti Zain, Z., & Soaad Syed Yahaya, S. (2015).
        """
        young = [482.43, 484.36, 488.84, 495.15, 495.24, 502.69, 504.62, 518.29, 519.1, 524.1, 524.12, 531.18, 548.42, 572.1, 584.68, 609.09, 609.53, 666.63, 676.4]
        middle = [335.59, 338.43, 353.54, 404.27, 437.5, 469.01, 485.85, 487.3, 493.08, 494.31, 499.1, 886.41]
        old = [519.01, 528.5, 530.23, 536.03, 538.56, 538.83, 557.24, 558.61, 558.95, 565.43, 586.39, 594.69, 629.22, 645.69, 691.84]
        soln = stats.alexandergovern(young, middle, old)
        assert_allclose(soln.statistic, 5.3237, atol=0.001)
        assert_allclose(soln.pvalue, 0.06982, atol=0.0001)
        '\n        > library("onewaytests")\n        > library("tibble")\n        > young <- c(482.43, 484.36, 488.84, 495.15, 495.24, 502.69, 504.62,\n        +                  518.29, 519.1, 524.1, 524.12, 531.18, 548.42, 572.1,\n        +                  584.68, 609.09, 609.53, 666.63, 676.4)\n        > middle <- c(335.59, 338.43, 353.54, 404.27, 437.5, 469.01, 485.85,\n        +                   487.3, 493.08, 494.31, 499.1, 886.41)\n        > old <- c(519.01, 528.5, 530.23, 536.03, 538.56, 538.83, 557.24,\n        +                   558.61, 558.95, 565.43, 586.39, 594.69, 629.22,\n        +                   645.69, 691.84)\n        > young_fct <- c(rep("young", times=19))\n        > middle_fct <-c(rep("middle", times=12))\n        > old_fct <- c(rep("old", times=15))\n        > ag.test(a ~ b, tibble(a=c(young, middle, old), b=factor(c(young_fct,\n        +                                              middle_fct, old_fct))))\n\n        Alexander-Govern Test (alpha = 0.05)\n        -------------------------------------------------------------\n        data : a and b\n\n        statistic  : 5.324629\n        parameter  : 2\n        p.value    : 0.06978651\n\n        Result     : Difference is not statistically significant.\n        -------------------------------------------------------------\n\n        '
        assert_allclose(soln.statistic, 5.324629)
        assert_allclose(soln.pvalue, 0.06978651)

    def test_compare_scholar3(self):
        """
        Data taken from 'Robustness And Comparative Power Of WelchAspin,
        Alexander-Govern And Yuen Tests Under Non-Normality And Variance
        Heteroscedasticity', by Ayed A. Almoied. 2017. Page 34-37.
        https://digitalcommons.wayne.edu/cgi/viewcontent.cgi?article=2775&context=oa_dissertations
        """
        x1 = [-1.77559, -1.4113, -0.69457, -0.54148, -0.18808, -0.07152, 0.04696, 0.051183, 0.148695, 0.168052, 0.422561, 0.458555, 0.616123, 0.709968, 0.839956, 0.857226, 0.929159, 0.981442, 0.999554, 1.642958]
        x2 = [-1.47973, -1.2722, -0.91914, -0.80916, -0.75977, -0.72253, -0.3601, -0.33273, -0.28859, -0.09637, -0.08969, -0.01824, 0.260131, 0.289278, 0.518254, 0.683003, 0.877618, 1.172475, 1.33964, 1.576766]
        soln = stats.alexandergovern(x1, x2)
        assert_allclose(soln.statistic, 0.713526, atol=1e-05)
        assert_allclose(soln.pvalue, 0.398276, atol=1e-05)
        '\n        tested in ag.test in R:\n        > library("onewaytests")\n        > library("tibble")\n        > x1 <- c(-1.77559, -1.4113, -0.69457, -0.54148, -0.18808, -0.07152,\n        +          0.04696, 0.051183, 0.148695, 0.168052, 0.422561, 0.458555,\n        +          0.616123, 0.709968, 0.839956, 0.857226, 0.929159, 0.981442,\n        +          0.999554, 1.642958)\n        > x2 <- c(-1.47973, -1.2722, -0.91914, -0.80916, -0.75977, -0.72253,\n        +         -0.3601, -0.33273, -0.28859, -0.09637, -0.08969, -0.01824,\n        +         0.260131, 0.289278, 0.518254, 0.683003, 0.877618, 1.172475,\n        +         1.33964, 1.576766)\n        > x1_fact <- c(rep("x1", times=20))\n        > x2_fact <- c(rep("x2", times=20))\n        > a <- c(x1, x2)\n        > b <- factor(c(x1_fact, x2_fact))\n        > ag.test(a ~ b, tibble(a, b))\n        Alexander-Govern Test (alpha = 0.05)\n        -------------------------------------------------------------\n        data : a and b\n\n        statistic  : 0.7135182\n        parameter  : 1\n        p.value    : 0.3982783\n\n        Result     : Difference is not statistically significant.\n        -------------------------------------------------------------\n        '
        assert_allclose(soln.statistic, 0.7135182)
        assert_allclose(soln.pvalue, 0.3982783)

    def test_nan_policy_propogate(self):
        args = [[1, 2, 3, 4], [1, np.nan]]
        res = stats.alexandergovern(*args)
        assert_equal(res.pvalue, np.nan)
        assert_equal(res.statistic, np.nan)

    def test_nan_policy_raise(self):
        args = [[1, 2, 3, 4], [1, np.nan]]
        with assert_raises(ValueError, match='The input contains nan values'):
            stats.alexandergovern(*args, nan_policy='raise')

    def test_nan_policy_omit(self):
        args_nan = [[1, 2, 3, None, 4], [1, np.nan, 19, 25]]
        args_no_nan = [[1, 2, 3, 4], [1, 19, 25]]
        res_nan = stats.alexandergovern(*args_nan, nan_policy='omit')
        res_no_nan = stats.alexandergovern(*args_no_nan)
        assert_equal(res_nan.pvalue, res_no_nan.pvalue)
        assert_equal(res_nan.statistic, res_no_nan.statistic)

    def test_constant_input(self):
        msg = 'An input array is constant; the statistic is not defined.'
        with assert_warns(stats.ConstantInputWarning, match=msg):
            res = stats.alexandergovern([0.667, 0.667, 0.667], [0.123, 0.456, 0.789])
            assert_equal(res.statistic, np.nan)
            assert_equal(res.pvalue, np.nan)