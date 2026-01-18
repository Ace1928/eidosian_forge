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
def _chk_datatype(klass, hdr, fix=False):
    rep = Report(HeaderDataError)
    code = int(hdr['datatype'])
    try:
        dtype = klass._data_type_codes.dtype[code]
    except KeyError:
        rep.problem_level = 40
        rep.problem_msg = 'data code %d not recognized' % code
    else:
        if dtype.itemsize == 0:
            rep.problem_level = 40
            rep.problem_msg = 'data code %d not supported' % code
        else:
            return (hdr, rep)
    if fix:
        rep.fix_msg = 'not attempting fix'
    return (hdr, rep)