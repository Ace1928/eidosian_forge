from __future__ import annotations
import numpy as np
from .arrayproxy import ArrayProxy
from .arraywriters import ArrayWriter, WriterError, get_slope_inter, make_array_writer
from .batteryrunners import Report
from .fileholders import copy_file_map
from .spatialimages import HeaderDataError, HeaderTypeError, SpatialHeader, SpatialImage
from .volumeutils import (
from .wrapstruct import LabeledWrapStruct
@classmethod
def _chk_sizeof_hdr(klass, hdr, fix=False):
    rep = Report(HeaderDataError)
    if hdr['sizeof_hdr'] == klass.sizeof_hdr:
        return (hdr, rep)
    rep.problem_level = 30
    rep.problem_msg = 'sizeof_hdr should be ' + str(klass.sizeof_hdr)
    if fix:
        hdr['sizeof_hdr'] = klass.sizeof_hdr
        rep.fix_msg = 'set sizeof_hdr to ' + str(klass.sizeof_hdr)
    return (hdr, rep)