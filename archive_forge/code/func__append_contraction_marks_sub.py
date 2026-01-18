import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def _append_contraction_marks_sub(Z, iv, i, n, contraction_marks, xp):
    if i >= n:
        contraction_marks.append((iv, Z[i - n, 2]))
        _append_contraction_marks_sub(Z, iv, int_floor(Z[i - n, 0], xp), n, contraction_marks, xp)
        _append_contraction_marks_sub(Z, iv, int_floor(Z[i - n, 1], xp), n, contraction_marks, xp)