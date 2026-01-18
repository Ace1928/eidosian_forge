import logging
from io import BytesIO, StringIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import imageglobals
from ..batteryrunners import Report
from ..casting import sctypes
from ..spatialimages import HeaderDataError
from ..volumeutils import Recoder, native_code, swapped_code
from ..wrapstruct import LabeledWrapStruct, WrapStruct, WrapStructError
@staticmethod
def _chk_string(hdr, fix=False):
    rep = Report(HeaderDataError)
    hdr_str = str(hdr['a_str'])
    if hdr_str.lower() == hdr_str:
        return (hdr, rep)
    rep.problem_level = 20
    rep.problem_msg = 'a_str should be lower case'
    if fix:
        hdr['a_str'] = hdr_str.lower()
        rep.fix_msg = 'set a_str to lower case'
    return (hdr, rep)