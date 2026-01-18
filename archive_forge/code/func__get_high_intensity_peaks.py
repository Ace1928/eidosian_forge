from warnings import warn
import numpy as np
import scipy.ndimage as ndi
from .. import measure
from .._shared.coord import ensure_spacing
def _get_high_intensity_peaks(image, mask, num_peaks, min_distance, p_norm):
    """
    Return the highest intensity peak coordinates.
    """
    coord = np.nonzero(mask)
    intensities = image[coord]
    idx_maxsort = np.argsort(-intensities, kind='stable')
    coord = np.transpose(coord)[idx_maxsort]
    if np.isfinite(num_peaks):
        max_out = int(num_peaks)
    else:
        max_out = None
    coord = ensure_spacing(coord, spacing=min_distance, p_norm=p_norm, max_out=max_out)
    if len(coord) > num_peaks:
        coord = coord[:num_peaks]
    return coord