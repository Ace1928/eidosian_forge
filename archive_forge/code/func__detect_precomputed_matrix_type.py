from . import api
from . import base
from . import graphs
from . import matrix
from . import utils
from functools import partial
from scipy import sparse
import abc
import numpy as np
import pygsp
import tasklogger
def _detect_precomputed_matrix_type(self, X):
    if isinstance(X, (sparse.coo_matrix, sparse.dia_matrix)):
        X = X.tocsr()
    if X[0, 0] == 0:
        return 'distance'
    else:
        return 'affinity'