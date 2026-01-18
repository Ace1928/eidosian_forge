from warnings import warn
import numpy as np
import scipy.ndimage as ndi
from .. import measure
from .._shared.coord import ensure_spacing
def _exclude_border(label, border_width):
    """Set label border values to 0."""
    for i, width in enumerate(border_width):
        if width == 0:
            continue
        label[(slice(None),) * i + (slice(None, width),)] = 0
        label[(slice(None),) * i + (slice(-width, None),)] = 0
    return label