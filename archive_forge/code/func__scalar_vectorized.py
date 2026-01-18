import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer
def _scalar_vectorized(scalar, M):
    """
    Scalar product between scalars and matrices.
    """
    return scalar[:, np.newaxis, np.newaxis] * M