import math
import textwrap
from abc import ABC, abstractmethod
import numpy as np
from scipy import spatial
from .._shared.utils import safe_as_int
from .._shared.compat import NP_COPY_IF_NEEDED
def _euler_rotation_matrix(angles, axes=None):
    """Produce an Euler rotation matrix from the given angles.

    The matrix will have dimension equal to the number of angles given.

    Parameters
    ----------
    angles : array of float, shape (3,)
        The transformation angles in radians.
    axes : list of int
        The axes about which to produce the rotation. Defaults to 0, 1, 2.

    Returns
    -------
    R : array of float, shape (3, 3)
        The Euler rotation matrix.
    """
    if axes is None:
        axes = range(3)
    dim = len(angles)
    R = np.eye(dim)
    for i, angle in zip(axes, angles):
        R = R @ _euler_rotation(i, angle)
    return R