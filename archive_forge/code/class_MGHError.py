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
class MGHError(Exception):
    """Exception for MGH format related problems.

    To be raised whenever MGH is not happy, or we are not happy with
    MGH.
    """