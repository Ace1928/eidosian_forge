import numpy as np
from .loadsave import load
from .orientations import OrientationError, io_orientation
def _aff_is_diag(aff):
    """Utility function returning True if affine is nearly diagonal"""
    rzs_aff = aff[:3, :3]
    return np.allclose(rzs_aff, np.diag(np.diag(rzs_aff)))