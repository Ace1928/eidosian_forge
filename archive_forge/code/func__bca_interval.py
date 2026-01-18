from __future__ import annotations
import warnings
import numpy as np
from itertools import combinations, permutations, product
from collections.abc import Sequence
import inspect
from scipy._lib._util import check_random_state, _rename_parameter
from scipy.special import ndtr, ndtri, comb, factorial
from scipy._lib._util import rng_integers
from dataclasses import dataclass
from ._common import ConfidenceInterval
from ._axis_nan_policy import _broadcast_concatenate, _broadcast_arrays
from ._warnings_errors import DegenerateDataWarning
def _bca_interval(data, statistic, axis, alpha, theta_hat_b, batch):
    """Bias-corrected and accelerated interval."""
    theta_hat = np.asarray(statistic(*data, axis=axis))[..., None]
    percentile = _percentile_of_score(theta_hat_b, theta_hat, axis=-1)
    z0_hat = ndtri(percentile)
    theta_hat_ji = []
    for j, sample in enumerate(data):
        samples = [np.expand_dims(sample, -2) for sample in data]
        theta_hat_i = []
        for jackknife_sample in _jackknife_resample(sample, batch):
            samples[j] = jackknife_sample
            broadcasted = _broadcast_arrays(samples, axis=-1)
            theta_hat_i.append(statistic(*broadcasted, axis=-1))
        theta_hat_ji.append(theta_hat_i)
    theta_hat_ji = [np.concatenate(theta_hat_i, axis=-1) for theta_hat_i in theta_hat_ji]
    n_j = [theta_hat_i.shape[-1] for theta_hat_i in theta_hat_ji]
    theta_hat_j_dot = [theta_hat_i.mean(axis=-1, keepdims=True) for theta_hat_i in theta_hat_ji]
    U_ji = [(n - 1) * (theta_hat_dot - theta_hat_i) for theta_hat_dot, theta_hat_i, n in zip(theta_hat_j_dot, theta_hat_ji, n_j)]
    nums = [(U_i ** 3).sum(axis=-1) / n ** 3 for U_i, n in zip(U_ji, n_j)]
    dens = [(U_i ** 2).sum(axis=-1) / n ** 2 for U_i, n in zip(U_ji, n_j)]
    a_hat = 1 / 6 * sum(nums) / sum(dens) ** (3 / 2)
    z_alpha = ndtri(alpha)
    z_1alpha = -z_alpha
    num1 = z0_hat + z_alpha
    alpha_1 = ndtr(z0_hat + num1 / (1 - a_hat * num1))
    num2 = z0_hat + z_1alpha
    alpha_2 = ndtr(z0_hat + num2 / (1 - a_hat * num2))
    return (alpha_1, alpha_2, a_hat)