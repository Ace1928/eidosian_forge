from io import BytesIO
import numpy as np
from packaging.version import Version, parse
from .. import xmlutils as xml
from ..batteryrunners import Report
from ..nifti1 import Nifti1Extension, extension_codes, intent_codes
from ..nifti2 import Nifti2Header, Nifti2Image
from ..spatialimages import HeaderDataError
from .cifti2 import (
class _Cifti2AsNiftiImage(Nifti2Image):
    """Load a NIfTI2 image with a Cifti2 header"""
    header_class = _Cifti2AsNiftiHeader
    makeable = False