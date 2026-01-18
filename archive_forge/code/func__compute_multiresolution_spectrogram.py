import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn import preprocessing, decomposition
import scprep
from . import utils
def _compute_multiresolution_spectrogram(self, sample_indicator):
    """Compute multiresolution spectrogram by repeatedly calling
        _compute_spectrogram"""
    spectrogram = np.zeros((self.windows[0].shape[1], self.eigenvectors.shape[1]))
    for window in self.windows:
        curr_spectrogram = self._compute_spectrogram(sample_indicator=sample_indicator, window=window)
        curr_spectrogram = self._activate(curr_spectrogram)
        spectrogram += curr_spectrogram
    return spectrogram