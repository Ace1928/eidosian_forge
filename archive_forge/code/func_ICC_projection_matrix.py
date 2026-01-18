import os
from functools import lru_cache
import numpy as np
from numpy import ones, kron, mean, eye, hstack, tile
from numpy.linalg import pinv
import nibabel as nb
from ..interfaces.base import (
@lru_cache(maxsize=1)
def ICC_projection_matrix(shape):
    nb_subjects, nb_conditions = shape
    x = kron(eye(nb_conditions), ones((nb_subjects, 1)))
    x0 = tile(eye(nb_subjects), (nb_conditions, 1))
    X = hstack([x, x0])
    return X @ pinv(X.T @ X, hermitian=True) @ X.T