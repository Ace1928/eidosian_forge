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
def __rqa_dp(sim: np.ndarray, gap_onset: float, gap_extend: float, knight: bool) -> Tuple[np.ndarray, np.ndarray]:
    """RQA dynamic programming implementation"""
    score = np.zeros(sim.shape, dtype=sim.dtype)
    backtrack = np.zeros(sim.shape, dtype=np.int8)
    sim_values = np.zeros(3)
    score_values = np.zeros(3)
    vec = np.zeros(3)
    if knight:
        init_limit = 2
        limit = 3
    else:
        init_limit = 1
        limit = 1
    score[0, :] = sim[0, :]
    score[:, 0] = sim[:, 0]
    for i in range(sim.shape[0]):
        if sim[i, 0]:
            backtrack[i, 0] = -2
        else:
            backtrack[i, 0] = -1
    for j in range(sim.shape[1]):
        if sim[0, j]:
            backtrack[0, j] = -2
        else:
            backtrack[0, j] = -1
    if sim[1, 1] > 0:
        score[1, 1] = score[0, 0] + sim[1, 1]
        backtrack[1, 1] = 0
    else:
        link = sim[0, 0] > 0
        score[1, 1] = max(0, score[0, 0] - link * gap_onset - ~link * gap_extend)
        if score[1, 1] > 0:
            backtrack[1, 1] = 0
        else:
            backtrack[1, 1] = -1
    i = 1
    for j in range(2, sim.shape[1]):
        score_values[:-1] = (score[i - 1, j - 1], score[i - 1, j - 2])
        sim_values[:-1] = (sim[i - 1, j - 1], sim[i - 1, j - 2])
        t_values = sim_values > 0
        if sim[i, j] > 0:
            backtrack[i, j] = np.argmax(score_values[:init_limit])
            score[i, j] = score_values[backtrack[i, j]] + sim[i, j]
        else:
            vec[:init_limit] = score_values[:init_limit] - t_values[:init_limit] * gap_onset - ~t_values[:init_limit] * gap_extend
            backtrack[i, j] = np.argmax(vec[:init_limit])
            score[i, j] = max(0, vec[backtrack[i, j]])
            if score[i, j] == 0:
                backtrack[i, j] = -1
    j = 1
    for i in range(2, sim.shape[0]):
        score_values[:-1] = (score[i - 1, j - 1], score[i - 2, j - 1])
        sim_values[:-1] = (sim[i - 1, j - 1], sim[i - 2, j - 1])
        t_values = sim_values > 0
        if sim[i, j] > 0:
            backtrack[i, j] = np.argmax(score_values[:init_limit])
            score[i, j] = score_values[backtrack[i, j]] + sim[i, j]
        else:
            vec[:init_limit] = score_values[:init_limit] - t_values[:init_limit] * gap_onset - ~t_values[:init_limit] * gap_extend
            backtrack[i, j] = np.argmax(vec[:init_limit])
            score[i, j] = max(0, vec[backtrack[i, j]])
            if score[i, j] == 0:
                backtrack[i, j] = -1
    for i in range(2, sim.shape[0]):
        for j in range(2, sim.shape[1]):
            score_values[:] = (score[i - 1, j - 1], score[i - 1, j - 2], score[i - 2, j - 1])
            sim_values[:] = (sim[i - 1, j - 1], sim[i - 1, j - 2], sim[i - 2, j - 1])
            t_values = sim_values > 0
            if sim[i, j] > 0:
                backtrack[i, j] = np.argmax(score_values[:limit])
                score[i, j] = score_values[backtrack[i, j]] + sim[i, j]
            else:
                vec[:limit] = score_values[:limit] - t_values[:limit] * gap_onset - ~t_values[:limit] * gap_extend
                backtrack[i, j] = np.argmax(vec[:limit])
                score[i, j] = max(0, vec[backtrack[i, j]])
                if score[i, j] == 0:
                    backtrack[i, j] = -1
    return (score, backtrack)