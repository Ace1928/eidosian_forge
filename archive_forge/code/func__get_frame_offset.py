import warnings
from numbers import Integral
import numpy as np
from .arraywriters import make_array_writer
from .fileslice import canonical_slicers, predict_shape, slice2outax
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import array_from_file, make_dt_codes, native_code, swapped_code
from .wrapstruct import WrapStruct
def _get_frame_offset(self, frame=0):
    return int(self._mlist[frame][1] * BLOCK_SIZE)