from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from numba import jit
from .util import pad_center, fill_off_diagonal, is_positive_int, tiny, expand_to
from .util.exceptions import ParameterError
from .filters import get_window
from typing import Any, Iterable, List, Optional, Tuple, Union, overload
from typing_extensions import Literal
from ._typing import _WindowSpec, _IntLike_co
@jit(nopython=True, cache=True)
def _viterbi(log_prob: np.ndarray, log_trans: np.ndarray, log_p_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Core Viterbi algorithm.

    This is intended for internal use only.

    Parameters
    ----------
    log_prob : np.ndarray [shape=(T, m)]
        ``log_prob[t, s]`` is the conditional log-likelihood
        ``log P[X = X(t) | State(t) = s]``
    log_trans : np.ndarray [shape=(m, m)]
        The log transition matrix
        ``log_trans[i, j] = log P[State(t+1) = j | State(t) = i]``
    log_p_init : np.ndarray [shape=(m,)]
        log of the initial state distribution

    Returns
    -------
    None
        All computations are performed in-place on ``state, value, ptr``.
    """
    n_steps, n_states = log_prob.shape
    state = np.zeros(n_steps, dtype=np.uint16)
    value = np.zeros((n_steps, n_states), dtype=np.float64)
    ptr = np.zeros((n_steps, n_states), dtype=np.uint16)
    value[0] = log_prob[0] + log_p_init
    for t in range(1, n_steps):
        trans_out = value[t - 1] + log_trans.T
        for j in range(n_states):
            ptr[t, j] = np.argmax(trans_out[j])
            value[t, j] = log_prob[t, j] + trans_out[j, ptr[t][j]]
    state[-1] = np.argmax(value[-1])
    for t in range(n_steps - 2, -1, -1):
        state[t] = ptr[t + 1, state[t + 1]]
    logp = value[-1:, state[-1]]
    return (state, logp)