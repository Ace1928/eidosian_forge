import time
import numpy as np
from tensorboard.plugins.scalar import metadata as scalars_metadata
from tensorboard.summary import _output
def _validate_scalar_shape(ndarray, name):
    if ndarray.ndim != 0:
        raise ValueError('Expected scalar value for %r but got %r' % (name, ndarray))
    return ndarray