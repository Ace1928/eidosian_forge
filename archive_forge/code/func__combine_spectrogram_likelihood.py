import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn import preprocessing, decomposition
import scprep
from . import utils
def _combine_spectrogram_likelihood(self, spectrogram, likelihood):
    """Normalizes and concatenates the likelihood to the
        spectrogram for clustering"""
    spectrogram_n = spectrogram / np.linalg.norm(spectrogram)
    ees_n = likelihood / np.linalg.norm(likelihood, ord=2, axis=0)
    ees_n = ees_n * self.likelihood_bias
    data_nu = np.c_[spectrogram_n, ees_n]
    return data_nu