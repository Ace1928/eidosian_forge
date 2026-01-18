from . import plot
from . import select
from . import utils
from ._lazyload import matplotlib
from scipy import sparse
from scipy import stats
from sklearn import metrics
from sklearn import neighbors
import joblib
import numbers
import numpy as np
import pandas as pd
import warnings
def EMD(x, y):
    """Compute Earth Mover's Distance between samples.

    Calculates an approximation of Earth Mover's Distance (also called
    Wasserstein distance) for 2 variables. This can be thought of as the
    distance between two probability distributions. This metric is useful for
    identifying differentially expressed genes between two groups of cells. For
    more information see https://en.wikipedia.org/wiki/Wasserstein_metric.

    Note, this is a wrapper function for scipy.stats.wasserstein_disance and assumes
    the data is 1-dimensional

    Parameters
    ----------
    x : array-like, shape=[n_samples]
        Input data (feature 1)
    y : array-like, shape=[n_samples]
        Input data (feature 2)

    Returns
    -------
    emd : float
        Earth Mover's Distance between x and y.

    Examples
    --------
    >>> import scprep
    >>> data = scprep.io.load_csv("my_data.csv")
    >>> emd = scprep.stats.EMD(data['GENE1'], data['GENE2'])
    """
    x, y = _vector_coerce_two_dense(x, y)
    return stats.wasserstein_distance(x, y)