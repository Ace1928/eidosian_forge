import math
import numpy as np
from contextlib import contextmanager
from matplotlib import (
from matplotlib.collections import (
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from . import proj3d
def _generate_normals(polygons):
    """
    Compute the normals of a list of polygons, one normal per polygon.

    Normals point towards the viewer for a face with its vertices in
    counterclockwise order, following the right hand rule.

    Uses three points equally spaced around the polygon. This method assumes
    that the points are in a plane. Otherwise, more than one shade is required,
    which is not supported.

    Parameters
    ----------
    polygons : list of (M_i, 3) array-like, or (..., M, 3) array-like
        A sequence of polygons to compute normals for, which can have
        varying numbers of vertices. If the polygons all have the same
        number of vertices and array is passed, then the operation will
        be vectorized.

    Returns
    -------
    normals : (..., 3) array
        A normal vector estimated for the polygon.
    """
    if isinstance(polygons, np.ndarray):
        n = polygons.shape[-2]
        i1, i2, i3 = (0, n // 3, 2 * n // 3)
        v1 = polygons[..., i1, :] - polygons[..., i2, :]
        v2 = polygons[..., i2, :] - polygons[..., i3, :]
    else:
        v1 = np.empty((len(polygons), 3))
        v2 = np.empty((len(polygons), 3))
        for poly_i, ps in enumerate(polygons):
            n = len(ps)
            i1, i2, i3 = (0, n // 3, 2 * n // 3)
            v1[poly_i, :] = ps[i1, :] - ps[i2, :]
            v2[poly_i, :] = ps[i2, :] - ps[i3, :]
    return np.cross(v1, v2)