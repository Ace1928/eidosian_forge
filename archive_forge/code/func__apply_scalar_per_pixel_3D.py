import numpy as np
from scipy import ndimage as ndi
from ..._shared.utils import check_nD, warn
from ...morphology.footprints import _footprint_is_sequence
from ...util import img_as_ubyte
from . import generic_cy
def _apply_scalar_per_pixel_3D(func, image, footprint, out, mask, shift_x, shift_y, shift_z, out_dtype=None):
    image, footprint, out, mask, n_bins = _handle_input_3D(image, footprint, out, mask, out_dtype, shift_x=shift_x, shift_y=shift_y, shift_z=shift_z)
    func(image, footprint, shift_x=shift_x, shift_y=shift_y, shift_z=shift_z, mask=mask, out=out, n_bins=n_bins)
    return out.reshape(out.shape[:3])