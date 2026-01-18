from *Random walks for image segmentation*, Leo Grady, IEEE Trans
import numpy as np
from scipy import sparse, ndimage as ndi
from .._shared import utils
from .._shared.utils import warn
from .._shared.compat import SCIPY_CG_TOL_PARAM_NAME
from ..util import img_as_float
from scipy.sparse.linalg import cg, spsolve
def _compute_weights_3d(data, spacing, beta, eps, multichannel):
    gradients = np.concatenate([np.diff(data[..., 0], axis=ax).ravel() / spacing[ax] for ax in [2, 1, 0] if data.shape[ax] > 1], axis=0) ** 2
    for channel in range(1, data.shape[-1]):
        gradients += np.concatenate([np.diff(data[..., channel], axis=ax).ravel() / spacing[ax] for ax in [2, 1, 0] if data.shape[ax] > 1], axis=0) ** 2
    scale_factor = -beta / (10 * data.std())
    if multichannel:
        scale_factor /= np.sqrt(data.shape[-1])
    weights = np.exp(scale_factor * gradients)
    weights += eps
    return -weights