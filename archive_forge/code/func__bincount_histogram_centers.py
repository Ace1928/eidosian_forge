import numpy as np
from ..util.dtype import dtype_range, dtype_limits
from .._shared import utils
def _bincount_histogram_centers(image, source_range):
    """Compute bin centers for bincount-based histogram."""
    if source_range not in ['image', 'dtype']:
        raise ValueError(f'Incorrect value for `source_range` argument: {source_range}')
    if source_range == 'image':
        image_min = int(image.min().astype(np.int64))
        image_max = int(image.max().astype(np.int64))
    elif source_range == 'dtype':
        image_min, image_max = dtype_limits(image, clip_negative=False)
    bin_centers = np.arange(image_min, image_max + 1)
    return bin_centers