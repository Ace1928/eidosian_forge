import numpy as np
import os
import scipy.ndimage
import scipy.spatial
from ..utils import *
def _hausdorff_distance(E_1, E_2):
    diamond = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    E_1_per = E_1 ^ scipy.ndimage.morphology.binary_erosion(E_1, structure=diamond)
    E_2_per = E_2 ^ scipy.ndimage.morphology.binary_erosion(E_2, structure=diamond)
    A = scipy.ndimage.morphology.distance_transform_edt(~E_2_per)[E_1_per].max()
    B = scipy.ndimage.morphology.distance_transform_edt(~E_1_per)[E_2_per].max()
    return np.max((A, B))