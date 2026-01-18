from itertools import permutations
import numpy as np
import math
from ._continuous_distns import norm
import scipy.stats
from dataclasses import dataclass
def _l_vectorized(ranks, predicted_ranks):
    """Calculate's Page's L statistic for each page of a 3d array"""
    colsums = ranks.sum(axis=-2, keepdims=True)
    products = predicted_ranks * colsums
    Ls = products.sum(axis=-1)
    Ls = Ls[0] if Ls.size == 1 else Ls.ravel()
    return Ls