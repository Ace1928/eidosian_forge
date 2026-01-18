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
@staticmethod
def _chk_offset(hdr, fix=False):
    rep = Report(HeaderDataError)
    magic = hdr['magic'].item()
    offset = hdr['vox_offset'].item()
    if offset == 0:
        return (hdr, rep)
    if magic == hdr.single_magic and offset < hdr.single_vox_offset:
        rep.problem_level = 40
        rep.problem_msg = 'vox offset %d too low for single file nifti1' % offset
        if fix:
            hdr['vox_offset'] = hdr.single_vox_offset
            rep.fix_msg = f'setting to minimum value of {hdr.single_vox_offset}'
        return (hdr, rep)
    if not offset % 16:
        return (hdr, rep)
    rep.problem_msg = f'vox offset (={offset:g}) not divisible by 16, not SPM compatible'
    rep.problem_level = 30
    if fix:
        rep.fix_msg = 'leaving at current value'
    return (hdr, rep)