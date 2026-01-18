import warnings
import numpy as np
from .tools import (
from .initialization import Initialization
from . import tools
def _set_vector(self, obj, value, shape):
    validate_vector_shape(self.name, value.shape, shape[0], obj.nobs)
    if value.ndim == 1:
        value = np.array(value[:, None], order='F')
    return value