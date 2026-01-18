import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from .._shared.utils import _supported_float_type
def _cv_init_level_set(init_level_set, image_shape, dtype=np.float64):
    """Generates an initial level set function conditional on input arguments."""
    if isinstance(init_level_set, str):
        if init_level_set == 'checkerboard':
            res = _cv_checkerboard(image_shape, 5, dtype)
        elif init_level_set == 'disk':
            res = _cv_large_disk(image_shape)
        elif init_level_set == 'small disk':
            res = _cv_small_disk(image_shape)
        else:
            raise ValueError('Incorrect name for starting level set preset.')
    else:
        res = init_level_set
    return res.astype(dtype, copy=False)