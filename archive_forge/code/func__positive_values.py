import os
import sys
import time
from itertools import zip_longest
import numpy as np
from numpy.testing import assert_
import pytest
from scipy.special._testutils import assert_func_equal
def _positive_values(self, a, b, n):
    if a < 0:
        raise ValueError('a should be positive')
    if n % 2 == 0:
        nlogpts = n // 2
        nlinpts = nlogpts
    else:
        nlogpts = n // 2
        nlinpts = nlogpts + 1
    if a >= 10:
        pts = np.logspace(np.log10(a), np.log10(b), n)
    elif a > 0 and b < 10:
        pts = np.linspace(a, b, n)
    elif a > 0:
        linpts = np.linspace(a, 10, nlinpts, endpoint=False)
        logpts = np.logspace(1, np.log10(b), nlogpts)
        pts = np.hstack((linpts, logpts))
    elif a == 0 and b <= 10:
        linpts = np.linspace(0, b, nlinpts)
        if linpts.size > 1:
            right = np.log10(linpts[1])
        else:
            right = -30
        logpts = np.logspace(-30, right, nlogpts, endpoint=False)
        pts = np.hstack((logpts, linpts))
    else:
        if nlogpts % 2 == 0:
            nlogpts1 = nlogpts // 2
            nlogpts2 = nlogpts1
        else:
            nlogpts1 = nlogpts // 2
            nlogpts2 = nlogpts1 + 1
        linpts = np.linspace(0, 10, nlinpts, endpoint=False)
        if linpts.size > 1:
            right = np.log10(linpts[1])
        else:
            right = -30
        logpts1 = np.logspace(-30, right, nlogpts1, endpoint=False)
        logpts2 = np.logspace(1, np.log10(b), nlogpts2)
        pts = np.hstack((logpts1, linpts, logpts2))
    return np.sort(pts)