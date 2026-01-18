import os
from copy import deepcopy
from nibabel import load, funcs, Nifti1Image
import numpy as np
from ..interfaces.base import (
from ..utils.filemanip import ensure_list, save_json, split_filename
from ..utils.misc import find_indices, normalize_mc_params
from .. import logging, config
def _get_affine_matrix(params, source):
    """Return affine matrix given a set of translation and rotation parameters

    params : np.array (up to 12 long) in native package format
    source : the package that generated the parameters
             supports SPM, AFNI, FSFAST, FSL, NIPY
    """
    if source == 'NIPY':
        from nipy.algorithms.registration import to_matrix44
        return to_matrix44(params)
    params = normalize_mc_params(params, source)
    rotfunc = lambda x: np.array([[np.cos(x), np.sin(x)], [-np.sin(x), np.cos(x)]])
    q = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    if len(params) < 12:
        params = np.hstack((params, q[len(params):]))
    params.shape = (len(params),)
    T = np.eye(4)
    T[0:3, -1] = params[0:3]
    Rx = np.eye(4)
    Rx[1:3, 1:3] = rotfunc(params[3])
    Ry = np.eye(4)
    Ry[(0, 0, 2, 2), (0, 2, 0, 2)] = rotfunc(params[4]).ravel()
    Rz = np.eye(4)
    Rz[0:2, 0:2] = rotfunc(params[5])
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(params[6:9])
    Sh = np.eye(4)
    Sh[(0, 0, 1), (1, 2, 2)] = params[9:12]
    if source in ('AFNI', 'FSFAST'):
        return np.dot(T, np.dot(Ry, np.dot(Rx, np.dot(Rz, np.dot(S, Sh)))))
    return np.dot(T, np.dot(Rx, np.dot(Ry, np.dot(Rz, np.dot(S, Sh)))))