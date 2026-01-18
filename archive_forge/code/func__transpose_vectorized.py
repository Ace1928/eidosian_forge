import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer
def _transpose_vectorized(M):
    """
    Transposition of an array of matrices *M*.
    """
    return np.transpose(M, [0, 2, 1])