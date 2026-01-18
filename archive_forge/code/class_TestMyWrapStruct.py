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
class TestMyWrapStruct(_TestWrapStructBase):
    """Test fake binary header defined at top of module"""
    header_class = MyWrapStruct

    def get_bad_bb(self):
        return b'\x00' * self.header_class.template_dtype.itemsize

    def _set_something_into_hdr(self, hdr):
        hdr['a_str'] = 'reggie'

    def test_empty(self):
        hdr = self.header_class()
        assert hdr['an_integer'] == 1
        assert hdr['a_str'] == b'a string'

    def test_str(self):
        hdr = self.header_class()
        s1 = str(hdr)
        assert len(s1) > 0
        assert 'an_integer' in s1
        assert 'a_str' in s1

    def test_copy(self):
        hdr = self.header_class()
        hdr2 = hdr.copy()
        assert hdr == hdr2
        self._set_something_into_hdr(hdr)
        assert hdr != hdr2
        self._set_something_into_hdr(hdr2)
        assert hdr == hdr2

    def test_copy(self):
        hdr = self.header_class()
        hdr2 = hdr.copy()
        assert hdr == hdr2
        self._set_something_into_hdr(hdr)
        assert hdr != hdr2
        self._set_something_into_hdr(hdr2)
        assert hdr == hdr2

    def test_checks(self):
        hdr_t = self.header_class()
        assert self._dxer(hdr_t) == ''
        hdr = hdr_t.copy()
        hdr['an_integer'] = 2
        assert self._dxer(hdr) == 'an_integer should be 1'
        hdr = hdr_t.copy()
        hdr['a_str'] = 'My Name'
        assert self._dxer(hdr) == 'a_str should be lower case'

    def test_log_checks(self):
        HC = self.header_class
        hdr = HC()
        hdr['an_integer'] = 2
        fhdr, message, raiser = self.log_chk(hdr, 40)
        return
        assert fhdr['an_integer'] == 1
        assert message == 'an_integer should be 1; set an_integer to 1'
        pytest.raises(*raiser)
        hdr = HC()
        hdr['a_str'] = 'Hello'
        fhdr, message, raiser = self.log_chk(hdr, 20)
        assert message == 'a_str should be lower case; set a_str to lower case'
        pytest.raises(*raiser)

    def test_logger_error(self):
        HC = self.header_class
        hdr = HC()
        str_io = StringIO()
        logger = logging.getLogger('test.logger')
        logger.setLevel(20)
        logger.addHandler(logging.StreamHandler(str_io))
        hdr['a_str'] = 'Fullness'
        log_cache = (imageglobals.logger, imageglobals.error_level)
        try:
            imageglobals.logger = logger
            hdr.copy().check_fix()
            assert str_io.getvalue() == 'a_str should be lower case; set a_str to lower case\n'
            imageglobals.error_level = 20
            with pytest.raises(HeaderDataError):
                hdr.copy().check_fix()
        finally:
            imageglobals.logger, imageglobals.error_level = log_cache