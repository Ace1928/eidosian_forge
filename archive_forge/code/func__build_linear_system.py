from *Random walks for image segmentation*, Leo Grady, IEEE Trans
import numpy as np
from scipy import sparse, ndimage as ndi
from .._shared import utils
from .._shared.utils import warn
from .._shared.compat import SCIPY_CG_TOL_PARAM_NAME
from ..util import img_as_float
from scipy.sparse.linalg import cg, spsolve
def _build_linear_system(data, spacing, labels, nlabels, mask, beta, multichannel):
    """
    Build the matrix A and rhs B of the linear system to solve.
    A and B are two block of the laplacian of the image graph.
    """
    if mask is None:
        labels = labels.ravel()
    else:
        labels = labels[mask]
    indices = np.arange(labels.size)
    seeds_mask = labels > 0
    unlabeled_indices = indices[~seeds_mask]
    seeds_indices = indices[seeds_mask]
    lap_sparse = _build_laplacian(data, spacing, mask=mask, beta=beta, multichannel=multichannel)
    rows = lap_sparse[unlabeled_indices, :]
    lap_sparse = rows[:, unlabeled_indices]
    B = -rows[:, seeds_indices]
    seeds = labels[seeds_mask]
    seeds_mask = sparse.csc_matrix(np.hstack([np.atleast_2d(seeds == lab).T for lab in range(1, nlabels + 1)]))
    rhs = B.dot(seeds_mask)
    return (lap_sparse, rhs)