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
class MyWrapStruct(WrapStruct):
    """An example wrapped struct class"""
    template_dtype = np.dtype([('an_integer', 'i2'), ('a_str', 'S10')])

    @classmethod
    def guessed_endian(klass, hdr):
        if hdr['an_integer'] < 256:
            return native_code
        return swapped_code

    @classmethod
    def default_structarr(klass, endianness=None):
        structarr = super().default_structarr(endianness)
        structarr['an_integer'] = 1
        structarr['a_str'] = b'a string'
        return structarr

    @classmethod
    def _get_checks(klass):
        """Return sequence of check functions for this class"""
        return (klass._chk_integer, klass._chk_string)
    ' Check functions in format expected by BatteryRunner class '

    @staticmethod
    def _chk_integer(hdr, fix=False):
        rep = Report(HeaderDataError)
        if hdr['an_integer'] == 1:
            return (hdr, rep)
        rep.problem_level = 40
        rep.problem_msg = 'an_integer should be 1'
        if fix:
            hdr['an_integer'] = 1
            rep.fix_msg = 'set an_integer to 1'
        return (hdr, rep)

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