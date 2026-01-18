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
def _build_kernel(self):
    """Private method to build kernel matrix

        Runs public method to build kernel matrix and runs
        additional checks to ensure that the result is okay

        Returns
        -------
        Kernel matrix, shape=[n_samples, n_samples]

        Raises
        ------
        RuntimeWarning : if K is not symmetric
        """
    kernel = self.build_kernel()
    kernel = self.symmetrize_kernel(kernel)
    kernel = self.apply_anisotropy(kernel)
    if (kernel - kernel.T).max() > 1e-05:
        warnings.warn('K should be symmetric', RuntimeWarning)
    if np.any(kernel.diagonal() == 0):
        warnings.warn('K should have a non-zero diagonal', RuntimeWarning)
    return kernel