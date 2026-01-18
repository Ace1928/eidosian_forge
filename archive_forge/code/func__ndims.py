from os.path import splitext
import numpy as np
from ..affines import from_matvec, voxel_sizes
from ..arrayproxy import ArrayProxy, reshape_dataobj
from ..batteryrunners import BatteryRunner, Report
from ..filebasedimages import SerializableImage
from ..fileholders import FileHolder
from ..filename_parser import _stringify_path
from ..openers import ImageOpener
from ..spatialimages import HeaderDataError, SpatialHeader, SpatialImage
from ..volumeutils import Recoder, array_from_file, array_to_file, endian_codes
from ..wrapstruct import LabeledWrapStruct
def _ndims(self):
    """Get dimensionality of data

        MGH does not encode dimensionality explicitly, so an image where the
        fourth dimension is 1 is treated as three-dimensional.

        Returns
        -------
        ndims : 3 or 4
        """
    return 3 + (self._structarr['dims'][3] > 1)