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
def _check_symmetrization(self, kernel_symm, theta):
    if kernel_symm not in ['+', '*', 'mnn', None]:
        raise ValueError("kernel_symm '{}' not recognized. Choose from '+', '*', 'mnn', or 'none'.".format(kernel_symm))
    elif kernel_symm != 'mnn' and theta is not None:
        warnings.warn("kernel_symm='{}' but theta is not None. Setting kernel_symm='mnn'.".format(kernel_symm))
        self.kernel_symm = kernel_symm = 'mnn'
    if kernel_symm == 'mnn':
        if theta is None:
            self.theta = theta = 1
            warnings.warn("kernel_symm='mnn' but theta not given. Defaulting to theta={}.".format(self.theta))
        elif not isinstance(theta, numbers.Number) or theta < 0 or theta > 1:
            raise ValueError('theta {} not recognized. Expected a float between 0 and 1'.format(theta))