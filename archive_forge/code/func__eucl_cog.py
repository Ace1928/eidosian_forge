import os
import os.path as op
import nibabel as nb
import numpy as np
from .. import config, logging
from ..interfaces.base import (
from ..interfaces.nipy.base import NipyBaseInterface
def _eucl_cog(self, nii1, nii2):
    from scipy.spatial.distance import cdist
    from scipy.ndimage.measurements import center_of_mass, label
    origdata1 = np.asanyarray(nii1.dataobj)
    origdata1 = (np.rint(origdata1) != 0) & ~np.isnan(origdata1)
    cog_t = np.array(center_of_mass(origdata1)).reshape(-1, 1)
    cog_t = np.vstack((cog_t, np.array([1])))
    cog_t_coor = np.dot(nii1.affine, cog_t)[:3, :]
    origdata2 = np.asanyarray(nii2.dataobj)
    origdata2 = (np.rint(origdata2) != 0) & ~np.isnan(origdata2)
    labeled_data, n_labels = label(origdata2)
    cogs = np.ones((4, n_labels))
    for i in range(n_labels):
        cogs[:3, i] = np.array(center_of_mass(origdata2, labeled_data, i + 1))
    cogs_coor = np.dot(nii2.affine, cogs)[:3, :]
    dist_matrix = cdist(cog_t_coor.T, cogs_coor.T)
    return np.mean(dist_matrix)