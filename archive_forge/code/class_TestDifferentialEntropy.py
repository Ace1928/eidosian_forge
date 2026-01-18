import numpy as np
from numpy.testing import assert_equal, assert_allclose
from numpy.testing import (assert_, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.stats as stats
class TestDifferentialEntropy:
    """
    Vasicek results are compared with the R package vsgoftest.

    # library(vsgoftest)
    #
    # samp <- c(<values>)
    # entropy.estimate(x = samp, window = <window_length>)

    """

    def test_differential_entropy_vasicek(self):
        random_state = np.random.RandomState(0)
        values = random_state.standard_normal(100)
        entropy = stats.differential_entropy(values, method='vasicek')
        assert_allclose(entropy, 1.342551, rtol=1e-06)
        entropy = stats.differential_entropy(values, window_length=1, method='vasicek')
        assert_allclose(entropy, 1.122044, rtol=1e-06)
        entropy = stats.differential_entropy(values, window_length=8, method='vasicek')
        assert_allclose(entropy, 1.349401, rtol=1e-06)

    def test_differential_entropy_vasicek_2d_nondefault_axis(self):
        random_state = np.random.RandomState(0)
        values = random_state.standard_normal((3, 100))
        entropy = stats.differential_entropy(values, axis=1, method='vasicek')
        assert_allclose(entropy, [1.342551, 1.341826, 1.293775], rtol=1e-06)
        entropy = stats.differential_entropy(values, axis=1, window_length=1, method='vasicek')
        assert_allclose(entropy, [1.122044, 1.102944, 1.129616], rtol=1e-06)
        entropy = stats.differential_entropy(values, axis=1, window_length=8, method='vasicek')
        assert_allclose(entropy, [1.349401, 1.338514, 1.292332], rtol=1e-06)

    def test_differential_entropy_raises_value_error(self):
        random_state = np.random.RandomState(0)
        values = random_state.standard_normal((3, 100))
        error_str = 'Window length \\({window_length}\\) must be positive and less than half the sample size \\({sample_size}\\).'
        sample_size = values.shape[1]
        for window_length in {-1, 0, sample_size // 2, sample_size}:
            formatted_error_str = error_str.format(window_length=window_length, sample_size=sample_size)
            with assert_raises(ValueError, match=formatted_error_str):
                stats.differential_entropy(values, window_length=window_length, axis=1)

    def test_base_differential_entropy_with_axis_0_is_equal_to_default(self):
        random_state = np.random.RandomState(0)
        values = random_state.standard_normal((100, 3))
        entropy = stats.differential_entropy(values, axis=0)
        default_entropy = stats.differential_entropy(values)
        assert_allclose(entropy, default_entropy)

    def test_base_differential_entropy_transposed(self):
        random_state = np.random.RandomState(0)
        values = random_state.standard_normal((3, 100))
        assert_allclose(stats.differential_entropy(values.T).T, stats.differential_entropy(values, axis=1))

    def test_input_validation(self):
        x = np.random.rand(10)
        message = '`base` must be a positive number or `None`.'
        with pytest.raises(ValueError, match=message):
            stats.differential_entropy(x, base=-2)
        message = '`method` must be one of...'
        with pytest.raises(ValueError, match=message):
            stats.differential_entropy(x, method='ekki-ekki')

    @pytest.mark.parametrize('method', ['vasicek', 'van es', 'ebrahimi', 'correa'])
    def test_consistency(self, method):
        n = 10000 if method == 'correa' else 1000000
        rvs = stats.norm.rvs(size=n, random_state=0)
        expected = stats.norm.entropy()
        res = stats.differential_entropy(rvs, method=method)
        assert_allclose(res, expected, rtol=0.005)
    norm_rmse_std_cases = {'vasicek': (0.198, 0.109), 'van es': (0.212, 0.11), 'correa': (0.135, 0.112), 'ebrahimi': (0.128, 0.109)}

    @pytest.mark.parametrize('method, expected', list(norm_rmse_std_cases.items()))
    def test_norm_rmse_std(self, method, expected):
        reps, n, m = (10000, 50, 7)
        rmse_expected, std_expected = expected
        rvs = stats.norm.rvs(size=(reps, n), random_state=0)
        true_entropy = stats.norm.entropy()
        res = stats.differential_entropy(rvs, window_length=m, method=method, axis=-1)
        assert_allclose(np.sqrt(np.mean((res - true_entropy) ** 2)), rmse_expected, atol=0.005)
        assert_allclose(np.std(res), std_expected, atol=0.002)
    expon_rmse_std_cases = {'vasicek': (0.194, 0.148), 'van es': (0.179, 0.149), 'correa': (0.155, 0.152), 'ebrahimi': (0.151, 0.148)}

    @pytest.mark.parametrize('method, expected', list(expon_rmse_std_cases.items()))
    def test_expon_rmse_std(self, method, expected):
        reps, n, m = (10000, 50, 7)
        rmse_expected, std_expected = expected
        rvs = stats.expon.rvs(size=(reps, n), random_state=0)
        true_entropy = stats.expon.entropy()
        res = stats.differential_entropy(rvs, window_length=m, method=method, axis=-1)
        assert_allclose(np.sqrt(np.mean((res - true_entropy) ** 2)), rmse_expected, atol=0.005)
        assert_allclose(np.std(res), std_expected, atol=0.002)

    @pytest.mark.parametrize('n, method', [(8, 'van es'), (12, 'ebrahimi'), (1001, 'vasicek')])
    def test_method_auto(self, n, method):
        rvs = stats.norm.rvs(size=(n,), random_state=0)
        res1 = stats.differential_entropy(rvs)
        res2 = stats.differential_entropy(rvs, method=method)
        assert res1 == res2