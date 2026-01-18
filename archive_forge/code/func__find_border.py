import os
import os.path as op
import nibabel as nb
import numpy as np
from .. import config, logging
from ..interfaces.base import (
from ..interfaces.nipy.base import NipyBaseInterface
def _find_border(self, data):
    from scipy.ndimage.morphology import binary_erosion
    eroded = binary_erosion(data)
    border = np.logical_and(data, np.logical_not(eroded))
    return border