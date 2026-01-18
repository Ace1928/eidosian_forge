import os
import os.path as op
import nibabel as nb
import numpy as np
from .. import config, logging
from ..interfaces.base import (
from ..interfaces.nipy.base import NipyBaseInterface
def _eucl_min(self, nii1, nii2):
    from scipy.spatial.distance import cdist, euclidean
    origdata1 = np.asanyarray(nii1.dataobj).astype(bool)
    border1 = self._find_border(origdata1)
    origdata2 = np.asanyarray(nii2.dataobj).astype(bool)
    border2 = self._find_border(origdata2)
    set1_coordinates = self._get_coordinates(border1, nii1.affine)
    set2_coordinates = self._get_coordinates(border2, nii2.affine)
    dist_matrix = cdist(set1_coordinates.T, set2_coordinates.T)
    point1, point2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    return (euclidean(set1_coordinates.T[point1, :], set2_coordinates.T[point2, :]), set1_coordinates.T[point1, :], set2_coordinates.T[point2, :])