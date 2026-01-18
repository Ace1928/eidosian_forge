import numpy as np
from functools import partial
from scipy import stats
def _bws_statistic(x, y, alternative, axis):
    """Compute the BWS test statistic for two independent samples"""
    Ri, Hj = (np.sort(x, axis=axis), np.sort(y, axis=axis))
    n, m = (Ri.shape[axis], Hj.shape[axis])
    i, j = (np.arange(1, n + 1), np.arange(1, m + 1))
    Bx_num = Ri - (m + n) / n * i
    By_num = Hj - (m + n) / m * j
    if alternative == 'two-sided':
        Bx_num *= Bx_num
        By_num *= By_num
    else:
        Bx_num *= np.abs(Bx_num)
        By_num *= np.abs(By_num)
    Bx_den = i / (n + 1) * (1 - i / (n + 1)) * m * (m + n) / n
    By_den = j / (m + 1) * (1 - j / (m + 1)) * n * (m + n) / m
    Bx = 1 / n * np.sum(Bx_num / Bx_den, axis=axis)
    By = 1 / m * np.sum(By_num / By_den, axis=axis)
    B = (Bx + By) / 2 if alternative == 'two-sided' else (Bx - By) / 2
    return B