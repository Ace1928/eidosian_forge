from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from ..exceptions import PlotnineError
from ..scales.scale_discrete import scale_discrete
def _adjust_breaks(breaks: FloatArray, right: bool) -> FloatArray:
    epsilon = np.finfo(float).eps
    plus = 1 + epsilon
    minus = 1 - epsilon
    sign = np.sign(breaks)
    pos_idx = np.where(sign == 1)[0]
    neg_idx = np.where(sign == -1)[0]
    zero_idx = np.where(sign == 0)[0]
    fuzzy = breaks.copy()
    if right:
        lbreak = breaks[0]
        fuzzy[pos_idx] *= plus
        fuzzy[neg_idx] *= minus
        fuzzy[zero_idx] = epsilon
        if lbreak == 0:
            fuzzy[0] = -epsilon
        elif lbreak < 0:
            fuzzy[0] = lbreak * plus
        else:
            fuzzy[0] = lbreak * minus
    else:
        rbreak = breaks[-1]
        fuzzy[pos_idx] *= minus
        fuzzy[neg_idx] *= plus
        fuzzy[zero_idx] = -epsilon
        if rbreak == 0:
            fuzzy[-1] = epsilon
        elif rbreak > 0:
            fuzzy[-1] = rbreak * plus
        else:
            fuzzy[-1] = rbreak * minus
    return fuzzy