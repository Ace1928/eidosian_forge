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
def _params_dunnett(samples: list[np.ndarray], control: np.ndarray) -> tuple[np.ndarray, int, int, np.ndarray, int]:
    """Specific parameters for Dunnett's test.

    Degree of freedom is the number of observations minus the number of groups
    including the control.
    """
    n_samples = np.array([sample.size for sample in samples])
    n_sample = n_samples.sum()
    n_control = control.size
    n = n_sample + n_control
    n_groups = len(samples)
    df = n - n_groups - 1
    rho = n_control / n_samples + 1
    rho = 1 / np.sqrt(rho[:, None] * rho[None, :])
    np.fill_diagonal(rho, 1)
    return (rho, df, n_groups, n_samples, n_control)