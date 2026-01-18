import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn import preprocessing, decomposition
import scprep
from . import utils
def _compute_windows(self):
    """_compute_window
        These windows mask the signal (sample_indicator) to perform a Windowed Graph
        Fourier Transform (WGFT) as described by Shuman et al.
        (https://arxiv.org/abs/1307.5708).

        This function is used when the power of windows is diadic and
        computes all windows efficiently.
        """
    windows = []
    curr_window = self._basewindow
    windows.append(preprocessing.normalize(curr_window, 'l2', axis=0).T)
    for i in range(len(self.window_sizes) - 1):
        curr_window = self._power_matrix(curr_window, 2)
        windows.append(preprocessing.normalize(curr_window, 'l2', axis=0).T)
    return windows