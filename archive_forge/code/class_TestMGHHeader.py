import io
import os
import pathlib
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from ... import imageglobals
from ...fileholders import FileHolder
from ...openers import ImageOpener
from ...spatialimages import HeaderDataError
from ...testing import data_path
from ...tests import test_spatialimages as tsi
from ...tests import test_wrapstruct as tws
from ...tmpdirs import InTemporaryDirectory
from ...volumeutils import sys_is_le
from ...wrapstruct import WrapStructError
from .. import load, save
from ..mghformat import MGHError, MGHHeader, MGHImage
class TestMGHHeader(tws._TestLabeledWrapStruct):
    header_class = MGHHeader

    def _set_something_into_hdr(self, hdr):
        hdr['dims'] = [4, 3, 2, 1]

    def get_bad_bb(self):
        return b'\xff' + b'\x00' * self.header_class._hdrdtype.itemsize

    def test_general_init(self):
        hdr = self.header_class()
        binblock = hdr.binaryblock
        assert len(binblock) == hdr.structarr.dtype.itemsize
        assert hdr.endianness == '>'
        hdr = self.header_class(check=False)

    def test__eq__(self):
        hdr1 = self.header_class()
        hdr2 = self.header_class()
        assert hdr1 == hdr2
        self._set_something_into_hdr(hdr1)
        assert hdr1 != hdr2
        self._set_something_into_hdr(hdr2)
        assert hdr1 == hdr2
        assert hdr1 != None
        assert hdr1 != 1

    def test_to_from_fileobj(self):
        hdr = self.header_class()
        str_io = io.BytesIO()
        hdr.write_to(str_io)
        str_io.seek(0)
        hdr2 = self.header_class.from_fileobj(str_io)
        assert hdr2.endianness == '>'
        assert hdr2.binaryblock == hdr.binaryblock

    def test_endian_guess(self):
        eh = self.header_class()
        assert eh.endianness == '>'
        assert self.header_class.guessed_endian(eh) == '>'

    def test_bytes(self):
        hdr1 = self.header_class()
        bb = hdr1.binaryblock
        hdr2 = self.header_class(hdr1.binaryblock)
        assert hdr1 == hdr2
        assert hdr1.binaryblock == hdr2.binaryblock
        self._set_something_into_hdr(hdr1)
        hdr2 = self.header_class(hdr1.binaryblock)
        assert hdr1 == hdr2
        assert hdr1.binaryblock == hdr2.binaryblock
        with pytest.raises(WrapStructError):
            self.header_class(bb[:self.header_class._hdrdtype.itemsize - 1])
        bb_bad = self.get_bad_bb()
        if bb_bad is None:
            return
        with imageglobals.LoggingOutputSuppressor():
            with pytest.raises(HeaderDataError):
                self.header_class(bb_bad)
        _ = self.header_class(bb_bad, check=False)

    def test_as_byteswapped(self):
        hdr = self.header_class()
        assert hdr.endianness == '>'
        for endianness in BIG_CODES:
            hdr2 = hdr.as_byteswapped(endianness)
            assert hdr2 is not hdr
            assert hdr2 == hdr
        for endianness in (None,) + LITTLE_CODES:
            with pytest.raises(ValueError):
                hdr.as_byteswapped(endianness)

        class DC(self.header_class):

            def check_fix(self, *args, **kwargs):
                raise Exception
        with pytest.raises(Exception):
            DC(hdr.binaryblock)
        hdr = DC(hdr.binaryblock, check=False)
        hdr2 = hdr.as_byteswapped('>')

    def test_checks(self):
        hdr_t = self.header_class()
        assert self._dxer(hdr_t) == ''
        hdr = hdr_t.copy()
        hdr['version'] = 2
        assert self._dxer(hdr) == 'Unknown MGH format version'