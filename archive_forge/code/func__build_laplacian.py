from *Random walks for image segmentation*, Leo Grady, IEEE Trans
import numpy as np
from scipy import sparse, ndimage as ndi
from .._shared import utils
from .._shared.utils import warn
from .._shared.compat import SCIPY_CG_TOL_PARAM_NAME
from ..util import img_as_float
from scipy.sparse.linalg import cg, spsolve
def _build_laplacian(data, spacing, mask, beta, multichannel):
    l_x, l_y, l_z = data.shape[:3]
    edges = _make_graph_edges_3d(l_x, l_y, l_z)
    weights = _compute_weights_3d(data, spacing, beta=beta, eps=1e-10, multichannel=multichannel)
    if mask is not None:
        mask0 = np.hstack([mask[..., :-1].ravel(), mask[:, :-1].ravel(), mask[:-1].ravel()])
        mask1 = np.hstack([mask[..., 1:].ravel(), mask[:, 1:].ravel(), mask[1:].ravel()])
        ind_mask = np.logical_and(mask0, mask1)
        edges, weights = (edges[:, ind_mask], weights[ind_mask])
        _, inv_idx = np.unique(edges, return_inverse=True)
        edges = inv_idx.reshape(edges.shape)
    pixel_nb = l_x * l_y * l_z
    i_indices = edges.ravel()
    j_indices = edges[::-1].ravel()
    data = np.hstack((weights, weights))
    lap = sparse.coo_matrix((data, (i_indices, j_indices)), shape=(pixel_nb, pixel_nb))
    lap.setdiag(-np.ravel(lap.sum(axis=0)))
    return lap.tocsr()