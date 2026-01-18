from . import matrix
from . import utils
from builtins import super
from copy import copy as shallow_copy
from future.utils import with_metaclass
from inspect import signature
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import abc
import numbers
import numpy as np
import pickle
import pygsp
import sys
import tasklogger
import warnings
def _check_extension_shape(self, Y):
    """Private method to check if new data matches `self.data`

        Parameters
        ----------
        Y : array-like, shape=[n_samples_y, n_features_y]
            Input data

        Returns
        -------
        Y : array-like, shape=[n_samples_y, n_pca]
            (Potentially transformed) input data

        Raises
        ------
        ValueError : if `n_features_y` is not either `self.data.shape[1]` or
        `self.n_pca`.
        """
    if len(Y.shape) != 2:
        raise ValueError('Expected a 2D matrix. Y has shape {}'.format(Y.shape))
    if not Y.shape[1] == self.data_nu.shape[1]:
        if Y.shape[1] == self.data.shape[1]:
            Y = self.transform(Y)
        else:
            if self.data.shape[1] != self.data_nu.shape[1]:
                msg = 'Y must be of shape either (n, {}) or (n, {})'.format(self.data.shape[1], self.data_nu.shape[1])
            else:
                msg = 'Y must be of shape (n, {})'.format(self.data.shape[1])
            raise ValueError(msg)
    return Y