import warnings
from numbers import Integral
import numpy as np
from .arraywriters import make_array_writer
from .fileslice import canonical_slicers, predict_shape, slice2outax
from .spatialimages import SpatialHeader, SpatialImage
from .volumeutils import array_from_file, make_dt_codes, native_code, swapped_code
from .wrapstruct import WrapStruct
def _get_empty_dir(self):
    """
        Get empty directory entry of the form
        [numAvail, nextDir, previousDir, numUsed]
        """
    return np.array([31, 2, 0, 0], dtype=np.int32)