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
def _check_data(self, data):
    if len(data.shape) != 2:
        msg = 'Expected 2D array, got {}D array instead (shape: {}.) '.format(len(data.shape), data.shape)
        if len(data.shape) < 2:
            msg += '\nReshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.'
        raise ValueError(msg)