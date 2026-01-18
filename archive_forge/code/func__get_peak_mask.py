from warnings import warn
import numpy as np
import scipy.ndimage as ndi
from .. import measure
from .._shared.coord import ensure_spacing
def _get_peak_mask(image, footprint, threshold, mask=None):
    """
    Return the mask containing all peak candidates above thresholds.
    """
    if footprint.size == 1 or image.size == 1:
        return image > threshold
    image_max = ndi.maximum_filter(image, footprint=footprint, mode='nearest')
    out = image == image_max
    image_is_trivial = np.all(out) if mask is None else np.all(out[mask])
    if image_is_trivial:
        out[:] = False
        if mask is not None:
            isolated_px = np.logical_xor(mask, ndi.binary_opening(mask))
            out[isolated_px] = True
    out &= image > threshold
    return out