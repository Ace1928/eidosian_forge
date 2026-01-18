from __future__ import division
from . import matrix
from . import utils
from .base import DataGraph
from .base import PyGSPGraph
from builtins import super
from scipy import sparse
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd
import numbers
import numpy as np
import tasklogger
import warnings
def _check_duplicates(self, distances, indices):
    if np.any(distances[:, 1] == 0):
        has_duplicates = distances[:, 1] == 0
        if np.sum(distances[:, 1:] == 0) < 20:
            idx = np.argwhere((distances == 0) & has_duplicates[:, None])
            duplicate_ids = np.array([[indices[i[0], i[1]], i[0]] for i in idx if indices[i[0], i[1]] < i[0]])
            duplicate_ids = duplicate_ids[np.argsort(duplicate_ids[:, 0])]
            duplicate_names = ', '.join(['{} and {}'.format(i[0], i[1]) for i in duplicate_ids])
            warnings.warn('Detected zero distance between samples {}. Consider removing duplicates to avoid errors in downstream processing.'.format(duplicate_names), RuntimeWarning)
        else:
            warnings.warn('Detected zero distance between {} pairs of samples. Consider removing duplicates to avoid errors in downstream processing.'.format(np.sum(np.sum(distances[:, 1:] == 0)) // 2), RuntimeWarning)