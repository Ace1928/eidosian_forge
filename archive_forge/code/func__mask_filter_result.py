import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_erosion, convolve
from .._shared.utils import _supported_float_type, check_nD
from ..restoration.uft import laplacian
from ..util.dtype import img_as_float
def _mask_filter_result(result, mask):
    """Return result after masking.

    Input masks are eroded so that mask areas in the original image don't
    affect values in the result.
    """
    if mask is not None:
        erosion_footprint = ndi.generate_binary_structure(mask.ndim, mask.ndim)
        mask = binary_erosion(mask, erosion_footprint, border_value=0)
        result *= mask
    return result