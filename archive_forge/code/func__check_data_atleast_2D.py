import math
from warnings import warn
import numpy as np
from numpy.linalg import inv
from scipy import optimize, spatial
def _check_data_atleast_2D(data):
    if data.ndim < 2 or data.shape[1] < 2:
        raise ValueError('Input data must be at least 2D.')