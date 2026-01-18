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
class TestMyLabeledWrapStruct(TestMyWrapStruct, _TestLabeledWrapStruct):
    header_class = MyLabeledWrapStruct

    def test_str(self):

        class MyHdr(self.header_class):
            _field_recoders = {}
        hdr = MyHdr()
        s1 = str(hdr)
        assert len(s1) > 0
        assert 'an_integer  : 1' in s1
        assert 'fullness of heart' not in s1
        rec = Recoder([[1, 'fullness of heart']], ('code', 'label'))
        hdr._field_recoders['an_integer'] = rec
        s2 = str(hdr)
        assert 'fullness of heart' in s2
        hdr['an_integer'] = 10
        s1 = str(hdr)
        assert '<unknown code 10>' in s1