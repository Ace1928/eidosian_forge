import os
from copy import deepcopy
from nibabel import load, funcs, Nifti1Image
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list, save_json, split_filename
from ..utils.misc import find_indices, normalize_mc_params
from .. import logging, config
def _calc_norm(mc, use_differences, source, brain_pts=None):
    """Calculates the maximum overall displacement of the midpoints
    of the faces of a cube due to translation and rotation.

    Parameters
    ----------
    mc : motion parameter estimates
        [3 translation, 3 rotation (radians)]
    use_differences : boolean
    brain_pts : [4 x n_points] of coordinates

    Returns
    -------

    norm : at each time point
    displacement : euclidean distance (mm) of displacement at each coordinate

    """
    affines = [_get_affine_matrix(mc[i, :], source) for i in range(mc.shape[0])]
    return _calc_norm_affine(affines, use_differences, brain_pts)