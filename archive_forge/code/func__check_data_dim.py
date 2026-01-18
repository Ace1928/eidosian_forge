import math
from warnings import warn
import numpy as np
from numpy.linalg import inv
from scipy import optimize, spatial
def _check_data_dim(data, dim):
    if data.ndim != 2 or data.shape[1] != dim:
        raise ValueError(f'Input data must have shape (N, {dim}).')