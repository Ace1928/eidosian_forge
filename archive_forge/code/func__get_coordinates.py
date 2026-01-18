import os
import os.path as op
import nibabel as nb
import numpy as np
from .. import config, logging
from ..interfaces.base import (
from ..interfaces.nipy.base import NipyBaseInterface
def _get_coordinates(self, data, affine):
    if len(data.shape) == 4:
        data = data[:, :, :, 0]
    indices = np.vstack(np.nonzero(data))
    indices = np.vstack((indices, np.ones(indices.shape[1])))
    coordinates = np.dot(affine, indices)
    return coordinates[:3, :]