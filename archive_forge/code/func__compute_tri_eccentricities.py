import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer
@staticmethod
def _compute_tri_eccentricities(tris_pts):
    """
        Compute triangle eccentricities.

        Parameters
        ----------
        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
            Coordinates of the triangles apexes.

        Returns
        -------
        array like of dim 2 (shape: (nx, 3))
            The so-called eccentricity parameters [1] needed for HCT triangular
            element.
        """
    a = np.expand_dims(tris_pts[:, 2, :] - tris_pts[:, 1, :], axis=2)
    b = np.expand_dims(tris_pts[:, 0, :] - tris_pts[:, 2, :], axis=2)
    c = np.expand_dims(tris_pts[:, 1, :] - tris_pts[:, 0, :], axis=2)
    dot_a = (_transpose_vectorized(a) @ a)[:, 0, 0]
    dot_b = (_transpose_vectorized(b) @ b)[:, 0, 0]
    dot_c = (_transpose_vectorized(c) @ c)[:, 0, 0]
    return _to_matrix_vectorized([[(dot_c - dot_b) / dot_a], [(dot_a - dot_c) / dot_b], [(dot_b - dot_a) / dot_c]])