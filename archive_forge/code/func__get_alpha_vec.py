import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer
@staticmethod
def _get_alpha_vec(x, y, tris_pts):
    """
        Fast (vectorized) function to compute barycentric coordinates alpha.

        Parameters
        ----------
        x, y : array-like of dim 1 (shape (nx,))
            Coordinates of the points whose points barycentric coordinates are
            requested.
        tris_pts : array like of dim 3 (shape: (nx, 3, 2))
            Coordinates of the containing triangles apexes.

        Returns
        -------
        array of dim 2 (shape (nx, 3))
            Barycentric coordinates of the points inside the containing
            triangles.
        """
    ndim = tris_pts.ndim - 2
    a = tris_pts[:, 1, :] - tris_pts[:, 0, :]
    b = tris_pts[:, 2, :] - tris_pts[:, 0, :]
    abT = np.stack([a, b], axis=-1)
    ab = _transpose_vectorized(abT)
    OM = np.stack([x, y], axis=1) - tris_pts[:, 0, :]
    metric = ab @ abT
    metric_inv = _pseudo_inv22sym_vectorized(metric)
    Covar = ab @ _transpose_vectorized(np.expand_dims(OM, ndim))
    ksi = metric_inv @ Covar
    alpha = _to_matrix_vectorized([[1 - ksi[:, 0, 0] - ksi[:, 1, 0]], [ksi[:, 0, 0]], [ksi[:, 1, 0]]])
    return alpha