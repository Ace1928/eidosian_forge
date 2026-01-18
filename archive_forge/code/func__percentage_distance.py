import numpy as np
import scipy.ndimage
import scipy.spatial
from ..motion.gme import globalEdgeMotion
from ..utils import *
def _percentage_distance(canny_in, canny_out, r):
    diamond = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    E_1 = scipy.ndimage.morphology.binary_dilation(canny_in, structure=diamond, iterations=r)
    E_2 = scipy.ndimage.morphology.binary_dilation(canny_out, structure=diamond, iterations=r)
    return 1.0 - np.float32(np.sum(E_1 & E_2)) / np.float32(np.sum(E_1))