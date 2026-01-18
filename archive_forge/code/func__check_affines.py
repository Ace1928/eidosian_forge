import warnings
from numbers import Integral
import numpy as np
from .arraywriters import make_array_writer
from .fileslice import canonical_slicers, predict_shape, slice2outax
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import array_from_file, make_dt_codes, native_code, swapped_code
from .wrapstruct import WrapStruct
def _check_affines(self):
    """checks if all affines are equal across frames"""
    nframes = self.get_nframes()
    if nframes == 1:
        return True
    affs = [self.get_frame_affine(i) for i in range(nframes)]
    if affs:
        i = iter(affs)
        first = next(i)
        for item in i:
            if not np.allclose(first, item):
                return False
    return True