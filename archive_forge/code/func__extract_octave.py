import numpy as np
from ..feature.util import (
from .corner import corner_fast, corner_orientations, corner_peaks, corner_harris
from ..transform import pyramid_gaussian
from .._shared.utils import check_nD
from .._shared.compat import NP_COPY_IF_NEEDED
from .orb_cy import _orb_loop
def _extract_octave(self, octave_image, keypoints, orientations):
    mask = _mask_border_keypoints(octave_image.shape, keypoints, distance=20)
    keypoints = np.array(keypoints[mask], dtype=np.intp, order='C', copy=NP_COPY_IF_NEEDED)
    orientations = np.array(orientations[mask], order='C', copy=False)
    descriptors = _orb_loop(octave_image, keypoints, orientations)
    return (descriptors, mask)