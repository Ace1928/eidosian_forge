import numpy as np
import pytest
from numpy.testing import assert_allclose
from skimage.draw import ellipsoid, ellipsoid_stats
from skimage.measure import marching_cubes, mesh_surface_area
def _same_mesh(vertices1, faces1, vertices2, faces2, tol=1e-10):
    """Compare two meshes, using a certain tolerance and invariant to
    the order of the faces.
    """
    triangles1 = vertices1[np.array(faces1)]
    triangles2 = vertices2[np.array(faces2)]
    triang1 = [np.concatenate(sorted(t, key=lambda x: tuple(x))) for t in triangles1]
    triang2 = [np.concatenate(sorted(t, key=lambda x: tuple(x))) for t in triangles2]
    triang1 = np.array(sorted([tuple(x) for x in triang1]))
    triang2 = np.array(sorted([tuple(x) for x in triang2]))
    return triang1.shape == triang2.shape and np.allclose(triang1, triang2, 0, tol)