from __future__ import annotations
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats._common import ConfidenceInterval
from scipy.stats._qmc import check_random_state
from scipy.stats._stats_py import _var
def _statistic_dunnett(samples: list[np.ndarray], control: np.ndarray, df: int, n_samples: np.ndarray, n_control: int) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Statistic of Dunnett's test.

    Computation based on the original single-step test from [1].
    """
    mean_control = np.mean(control)
    mean_samples = np.array([np.mean(sample) for sample in samples])
    all_samples = [control] + samples
    all_means = np.concatenate([[mean_control], mean_samples])
    s2 = np.sum([_var(sample, mean=mean) * sample.size for sample, mean in zip(all_samples, all_means)]) / df
    std = np.sqrt(s2)
    z = (mean_samples - mean_control) / np.sqrt(1 / n_samples + 1 / n_control)
    return (z / std, std, mean_control, mean_samples)