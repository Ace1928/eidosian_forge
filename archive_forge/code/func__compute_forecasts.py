import contextlib
from warnings import warn
import numpy as np
from .representation import OptionWrapper, Representation, FrozenRepresentation
from .tools import reorder_missing_matrix, reorder_missing_vector
from . import tools
from statsmodels.tools.sm_exceptions import ValueWarning
def _compute_forecasts(self, states, states_cov, signal_only=False):
    d = self.obs_intercept
    Z = self.design
    H = self.obs_cov
    if d.ndim == 1:
        d = d[:, None]
    if Z.ndim == 2:
        if not signal_only:
            forecasts = d + Z @ states
            forecasts_error_cov = (Z[None, ...] @ states_cov.T @ Z.T[None, ...] + H.T).T
        else:
            forecasts = Z @ states
            forecasts_error_cov = (Z[None, ...] @ states_cov.T @ Z.T[None, ...]).T
    elif not signal_only:
        forecasts = d + (Z * states[None, :, :]).sum(axis=1)
        tmp = Z[:, None, ...] * states_cov[None, ...]
        tmp = tmp[:, :, :, None, :] * Z.transpose(1, 0, 2)[None, :, None, ...]
        forecasts_error_cov = (tmp.sum(axis=1).sum(axis=1).T + H.T).T
    else:
        forecasts = (Z * states[None, :, :]).sum(axis=1)
        tmp = Z[:, None, ...] * states_cov[None, ...]
        tmp = tmp[:, :, :, None, :] * Z.transpose(1, 0, 2)[None, :, None, ...]
        forecasts_error_cov = tmp.sum(axis=1).sum(axis=1)
    return (forecasts, forecasts_error_cov)