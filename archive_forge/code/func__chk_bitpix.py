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
def _chk_bitpix(klass, hdr, fix=False):
    rep = Report(HeaderDataError)
    code = int(hdr['datatype'])
    try:
        dt = klass._data_type_codes.dtype[code]
    except KeyError:
        rep.problem_level = 10
        rep.problem_msg = 'no valid datatype to fix bitpix'
        if fix:
            rep.fix_msg = 'no way to fix bitpix'
        return (hdr, rep)
    bitpix = dt.itemsize * 8
    if bitpix == hdr['bitpix']:
        return (hdr, rep)
    rep.problem_level = 10
    rep.problem_msg = 'bitpix does not match datatype'
    if fix:
        hdr['bitpix'] = bitpix
        rep.fix_msg = 'setting bitpix to match datatype'
    return (hdr, rep)