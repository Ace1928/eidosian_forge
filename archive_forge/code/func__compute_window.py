import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn import preprocessing, decomposition
import scprep
from . import utils
def _compute_window(self, window, t=1):
    """_compute_window
        These windows mask the signal (sample_indicator) to perform a Windowed Graph
        Fourier Transform (WGFT) as described by Shuman et al.
        (https://arxiv.org/abs/1307.5708).

        This function is used when the power of windows is NOT diadic
        """
    if sparse.issparse(window):
        window = window ** t
    else:
        window = np.linalg.matrix_power(window, t)
    return preprocessing.normalize(window, 'l2', axis=0).T