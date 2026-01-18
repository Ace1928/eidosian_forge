import os
from copy import deepcopy
from nibabel import load, funcs, Nifti1Image
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list, save_json, split_filename
from ..utils.misc import find_indices, normalize_mc_params
from .. import logging, config
def _calc_norm_affine(affines, use_differences, brain_pts=None):
    """Calculates the maximum overall displacement of the midpoints
    of the faces of a cube due to translation and rotation.

    Parameters
    ----------
    affines : list of [4 x 4] affine matrices
    use_differences : boolean
    brain_pts : [4 x n_points] of coordinates

    Returns
    -------

    norm : at each time point
    displacement : euclidean distance (mm) of displacement at each coordinate

    """
    if brain_pts is None:
        respos = np.diag([70, 70, 75])
        resneg = np.diag([-70, -110, -45])
        all_pts = np.vstack((np.hstack((respos, resneg)), np.ones((1, 6))))
        displacement = None
    else:
        all_pts = brain_pts
    n_pts = all_pts.size - all_pts.shape[1]
    newpos = np.zeros((len(affines), n_pts))
    if brain_pts is not None:
        displacement = np.zeros((len(affines), int(n_pts / 3)))
    for i, affine in enumerate(affines):
        newpos[i, :] = np.dot(affine, all_pts)[0:3, :].ravel()
        if brain_pts is not None:
            displacement[i, :] = np.sqrt(np.sum(np.power(np.reshape(newpos[i, :], (3, all_pts.shape[1])) - all_pts[0:3, :], 2), axis=0))
    normdata = np.zeros(len(affines))
    if use_differences:
        newpos = np.concatenate((np.zeros((1, n_pts)), np.diff(newpos, n=1, axis=0)), axis=0)
        for i in range(newpos.shape[0]):
            normdata[i] = np.max(np.sqrt(np.sum(np.reshape(np.power(np.abs(newpos[i, :]), 2), (3, all_pts.shape[1])), axis=0)))
    else:
        from scipy.signal import detrend
        newpos = np.abs(detrend(newpos, axis=0, type='constant'))
        normdata = np.sqrt(np.mean(np.power(newpos, 2), axis=1))
    return (normdata, displacement)