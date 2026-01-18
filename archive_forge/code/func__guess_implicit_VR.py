from __future__ import annotations
import warnings
from io import BytesIO
import numpy as np
import numpy.linalg as npl
from . import analyze  # module import
from .arrayproxy import get_obj_dtype
from .batteryrunners import Report
from .casting import have_binary128
from .deprecated import alert_future_error
from .filebasedimages import ImageFileError, SerializableImage
from .optpkg import optional_package
from .quaternions import fillpositive, mat2quat, quat2mat
from .spatialimages import HeaderDataError
from .spm99analyze import SpmAnalyzeHeader
from .volumeutils import Recoder, endian_codes, make_dt_codes
def _guess_implicit_VR(self):
    """Try to guess DICOM syntax by checking for valid VRs.

        Without a DICOM Transfer Syntax, it's difficult to tell if Value
        Representations (VRs) are included in the DICOM encoding or not.
        This reads where the first VR would be and checks it against a list of
        valid VRs
        """
    potential_vr = self._raw_content[4:6].decode()
    if potential_vr in pdcm.values.converters.keys():
        implicit_VR = False
    else:
        implicit_VR = True
    return implicit_VR