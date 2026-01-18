from itertools import cycle
import numpy as np
from scipy import ndimage as ndi
from .._shared.utils import check_nD
def _init_level_set(init_level_set, image_shape):
    """Auxiliary function for initializing level sets with a string.

    If `init_level_set` is not a string, it is returned as is.
    """
    if isinstance(init_level_set, str):
        if init_level_set == 'checkerboard':
            res = checkerboard_level_set(image_shape)
        elif init_level_set == 'disk':
            res = disk_level_set(image_shape)
        else:
            raise ValueError("`init_level_set` not in ['checkerboard', 'disk']")
    else:
        res = init_level_set
    return res