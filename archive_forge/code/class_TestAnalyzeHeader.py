import itertools
import logging
import os
import pickle
import re
from io import BytesIO, StringIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import imageglobals
from ..analyze import AnalyzeHeader, AnalyzeImage
from ..arraywriters import WriterError
from ..casting import sctypes_aliases
from ..nifti1 import Nifti1Header
from ..optpkg import optional_package
from ..spatialimages import HeaderDataError, HeaderTypeError, supported_np_types
from ..testing import (
from ..tmpdirs import InTemporaryDirectory
from . import test_spatialimages as tsi
from . import test_wrapstruct as tws
class TestAnalyzeHeader(tws._TestLabeledWrapStruct):
    header_class = AnalyzeHeader
    example_file = header_file
    sizeof_hdr = AnalyzeHeader.sizeof_hdr
    supported_np_types = {np.uint8, np.int16, np.int32, np.float32, np.float64, np.complex64}
    add_duplicate_types(supported_np_types)

    def test_supported_types(self):
        hdr = self.header_class()
        assert self.supported_np_types == supported_np_types(hdr)

    def get_bad_bb(self):
        return b'\x00' * self.header_class.template_dtype.itemsize

    def test_general_init(self):
        super().test_general_init()
        hdr = self.header_class()
        assert hdr.get_data_shape() == (0,)
        assert_array_equal(np.diag(hdr.get_base_affine()), [-1, 1, 1, 1])
        assert hdr.get_zooms() == (1.0,)

    def test_header_size(self):
        assert self.header_class.template_dtype.itemsize == self.sizeof_hdr

    def test_empty(self):
        hdr = self.header_class()
        assert len(hdr.binaryblock) == self.sizeof_hdr
        assert hdr['sizeof_hdr'] == self.sizeof_hdr
        assert np.all(hdr['dim'][1:] == 1)
        assert hdr['dim'][0] == 0
        assert np.all(hdr['pixdim'] == 1)
        assert hdr['datatype'] == 16
        assert hdr['bitpix'] == 32

    def _set_something_into_hdr(self, hdr):
        with suppress_warnings():
            hdr.set_data_shape((1, 2, 3))

    def test_checks(self):
        hdr_t = self.header_class()
        assert self._dxer(hdr_t) == ''
        hdr = hdr_t.copy()
        hdr['sizeof_hdr'] = 1
        with suppress_warnings():
            assert self._dxer(hdr) == 'sizeof_hdr should be ' + str(self.sizeof_hdr)
        hdr = hdr_t.copy()
        hdr['datatype'] = 0
        assert self._dxer(hdr) == 'data code 0 not supported\nbitpix does not match datatype'
        hdr = hdr_t.copy()
        hdr['bitpix'] = 0
        assert self._dxer(hdr) == 'bitpix does not match datatype'

    def test_pixdim_checks(self):
        hdr_t = self.header_class()
        for i in (1, 2, 3):
            hdr = hdr_t.copy()
            hdr['pixdim'][i] = -1
            assert self._dxer(hdr) == 'pixdim[1,2,3] should be positive'

    def test_log_checks(self):
        HC = self.header_class
        hdr = HC()
        with suppress_warnings():
            hdr['sizeof_hdr'] = 350
            fhdr, message, raiser = self.log_chk(hdr, 30)
        assert fhdr['sizeof_hdr'] == self.sizeof_hdr
        assert message == f'sizeof_hdr should be {self.sizeof_hdr}; set sizeof_hdr to {self.sizeof_hdr}'
        pytest.raises(*raiser)
        hdr = HC()
        hdr.set_data_dtype('RGB')
        fhdr, message, raiser = self.log_chk(hdr, 0)
        hdr = HC()
        hdr['datatype'] = -1
        with suppress_warnings():
            fhdr, message, raiser = self.log_chk(hdr, 40)
        assert message == 'data code -1 not recognized; not attempting fix'
        pytest.raises(*raiser)
        hdr['datatype'] = 255
        fhdr, message, raiser = self.log_chk(hdr, 40)
        assert message == 'data code 255 not supported; not attempting fix'
        pytest.raises(*raiser)
        hdr = HC()
        hdr['datatype'] = 16
        hdr['bitpix'] = 16
        fhdr, message, raiser = self.log_chk(hdr, 10)
        assert fhdr['bitpix'] == 32
        assert message == 'bitpix does not match datatype; setting bitpix to match datatype'
        pytest.raises(*raiser)

    def test_pixdim_log_checks(self):
        HC = self.header_class
        hdr = HC()
        hdr['pixdim'][1] = -2
        fhdr, message, raiser = self.log_chk(hdr, 35)
        assert fhdr['pixdim'][1] == 2
        assert message == 'pixdim[1,2,3] should be positive; setting to abs of pixdim values'
        pytest.raises(*raiser)
        hdr = HC()
        hdr['pixdim'][1] = 0
        fhdr, message, raiser = self.log_chk(hdr, 30)
        assert fhdr['pixdim'][1] == 1
        assert message == PIXDIM0_MSG
        pytest.raises(*raiser)
        hdr = HC()
        hdr['pixdim'][1] = 0
        hdr['pixdim'][2] = -2
        fhdr, message, raiser = self.log_chk(hdr, 35)
        assert fhdr['pixdim'][1] == 1
        assert fhdr['pixdim'][2] == 2
        assert message == 'pixdim[1,2,3] should be non-zero and pixdim[1,2,3] should be positive; setting 0 dims to 1 and setting to abs of pixdim values'
        pytest.raises(*raiser)

    def test_no_scaling_fixes(self):
        HC = self.header_class
        if not HC.has_data_slope:
            return
        hdr = HC()
        has_inter = HC.has_data_intercept
        slopes = (1, 0, np.nan, np.inf, -np.inf)
        inters = (0, np.nan, np.inf, -np.inf) if has_inter else (0,)
        for slope, inter in itertools.product(slopes, inters):
            hdr['scl_slope'] = slope
            if has_inter:
                hdr['scl_inter'] = inter
            self.assert_no_log_err(hdr)

    def test_logger_error(self):
        HC = self.header_class
        hdr = HC()
        str_io = StringIO()
        logger = logging.getLogger('test.logger')
        logger.addHandler(logging.StreamHandler(str_io))
        hdr['datatype'] = 16
        hdr['bitpix'] = 16
        logger.setLevel(10)
        log_cache = (imageglobals.logger, imageglobals.error_level)
        try:
            imageglobals.logger = logger
            hdr.copy().check_fix()
            assert str_io.getvalue() == 'bitpix does not match datatype; setting bitpix to match datatype\n'
            imageglobals.error_level = 10
            with pytest.raises(HeaderDataError):
                hdr.copy().check_fix()
        finally:
            imageglobals.logger, imageglobals.error_level = log_cache

    def test_data_dtype(self):
        all_supported_types = ((2, np.uint8), (4, np.int16), (8, np.int32), (16, np.float32), (32, np.complex64), (64, np.float64), (128, np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])))
        all_unsupported_types = (np.void, 'none', 'all', 0)

        def assert_set_dtype(dt_spec, np_dtype):
            hdr = self.header_class()
            hdr.set_data_dtype(dt_spec)
            assert_dt_equal(hdr.get_data_dtype(), np_dtype)
        for code, npt in all_supported_types:
            assert_set_dtype(code, npt)
            assert_set_dtype(npt, npt)
            assert_set_dtype(np.dtype(npt), npt)
        for npt in self.supported_np_types:
            assert_set_dtype(npt, npt)
            assert_set_dtype(np.dtype(npt), npt)
            assert_set_dtype(np.dtype(npt).newbyteorder(), npt)
            assert_set_dtype(np.dtype(npt).str, npt)
            if np.dtype(npt).str[0] in '=|<>':
                assert_set_dtype(np.dtype(npt).str[1:], npt)
        assert_set_dtype(float, np.float64)
        np_sys_int = np.dtype(int).type
        if issubclass(self.header_class, Nifti1Header):
            with pytest.raises(ValueError):
                hdr = self.header_class()
                hdr.set_data_dtype(int)
        elif np_sys_int in self.supported_np_types:
            assert_set_dtype(int, np_sys_int)
        hdr = self.header_class()
        for inp in all_unsupported_types:
            with pytest.raises(HeaderDataError):
                hdr.set_data_dtype(inp)

    def test_shapes(self):
        hdr = self.header_class()
        for shape in ((2, 3, 4), (2, 3, 4, 5), (2, 3), (2,)):
            hdr.set_data_shape(shape)
            assert hdr.get_data_shape() == shape
        dim_dtype = hdr.structarr['dim'].dtype
        mx = int(np.iinfo(dim_dtype).max)
        shape = (mx,)
        hdr.set_data_shape(shape)
        assert hdr.get_data_shape() == shape
        shape = (mx + 1,)
        with pytest.raises(HeaderDataError):
            hdr.set_data_shape(shape)
        shape = (2, 3, 4)
        for constructor in (list, tuple, np.array):
            hdr.set_data_shape(constructor(shape))
            assert hdr.get_data_shape() == shape

    def test_read_write_data(self):
        hdr = self.header_class()
        bytes = hdr.data_from_fileobj(BytesIO())
        assert len(bytes) == 0
        str_io = BytesIO()
        hdr.data_to_fileobj([], str_io)
        assert str_io.getvalue() == b''
        with pytest.raises(HeaderDataError):
            hdr.data_to_fileobj(np.zeros(3), str_io)
        hdr.set_data_shape((1, 2, 3))
        hdr.set_data_dtype(np.float32)
        S = BytesIO()
        data = np.arange(6, dtype=np.float64)
        with pytest.raises(HeaderDataError):
            hdr.data_to_fileobj(data, S)
        data = data.reshape((1, 2, 3))
        with pytest.raises(HeaderDataError):
            hdr.data_to_fileobj(data[:, :, :-1], S)
        with pytest.raises(HeaderDataError):
            hdr.data_to_fileobj(data[:, :-1, :], S)
        hdr.data_to_fileobj(data, S)
        data_back = hdr.data_from_fileobj(S)
        assert_array_almost_equal(data, data_back)
        assert hdr.get_data_dtype() == data_back.dtype
        S2 = BytesIO()
        hdr2 = hdr.as_byteswapped()
        hdr2.set_data_dtype(np.float32)
        hdr2.set_data_shape((1, 2, 3))
        hdr2.data_to_fileobj(data, S2)
        data_back2 = hdr2.data_from_fileobj(S2)
        assert_array_almost_equal(data_back, data_back2)
        assert data_back.dtype.name == data_back2.dtype.name
        assert data.dtype.byteorder != data_back2.dtype.byteorder
        hdr.set_data_dtype(np.uint8)
        S3 = BytesIO()
        with np.errstate(invalid='ignore'):
            hdr.data_to_fileobj(data, S3, rescale=False)
        data_back = hdr.data_from_fileobj(S3)
        assert_array_almost_equal(data, data_back)
        if not hdr.has_data_slope:
            with pytest.raises(HeaderTypeError):
                hdr.data_to_fileobj(data, S3)
            with pytest.raises(HeaderTypeError):
                hdr.data_to_fileobj(data, S3, rescale=True)
        data = np.arange(6, dtype=np.float64).reshape((1, 2, 3)) + 0.5
        with np.errstate(invalid='ignore'):
            hdr.data_to_fileobj(data, S3, rescale=False)
        data_back = hdr.data_from_fileobj(S3)
        assert not np.allclose(data, data_back)
        dtype = np.dtype([('R', 'uint8'), ('G', 'uint8'), ('B', 'uint8')])
        data = np.ones((1, 2, 3), dtype)
        hdr.set_data_dtype(dtype)
        S4 = BytesIO()
        hdr.data_to_fileobj(data, S4)
        data_back = hdr.data_from_fileobj(S4)
        assert_array_equal(data, data_back)

    def test_datatype(self):
        ehdr = self.header_class()
        codes = self.header_class._data_type_codes
        for code in codes.value_set():
            npt = codes.type[code]
            if npt is np.void:
                with pytest.raises(HeaderDataError):
                    ehdr.set_data_dtype(code)
                continue
            dt = codes.dtype[code]
            ehdr.set_data_dtype(npt)
            assert ehdr['datatype'] == code
            assert ehdr['bitpix'] == dt.itemsize * 8
            ehdr.set_data_dtype(code)
            assert ehdr['datatype'] == code
            ehdr.set_data_dtype(dt)
            assert ehdr['datatype'] == code

    def test_offset(self):
        hdr = self.header_class()
        offset = hdr.get_data_offset()
        hdr.set_data_offset(offset + 16)
        assert hdr.get_data_offset() == offset + 16

    def test_data_shape_zooms_affine(self):
        hdr = self.header_class()
        for shape in ((1, 2, 3), (0,), (1,), (1, 2), (1, 2, 3, 4)):
            L = len(shape)
            hdr.set_data_shape(shape)
            if L:
                assert hdr.get_data_shape() == shape
            else:
                assert hdr.get_data_shape() == (0,)
            assert hdr.get_zooms() == (1,) * L
            if len(shape):
                with pytest.raises(HeaderDataError):
                    hdr.set_zooms((1,) * (L - 1))
                with pytest.raises(HeaderDataError):
                    hdr.set_zooms((-1,) + (1,) * (L - 1))
            with pytest.raises(HeaderDataError):
                hdr.set_zooms((1,) * (L + 1))
            with pytest.raises(HeaderDataError):
                hdr.set_zooms((-1,) * L)
        hdr = self.header_class()
        hdr.set_data_shape((1, 2, 3))
        hdr.set_zooms((4, 5, 6))
        assert_array_equal(hdr.get_zooms(), (4, 5, 6))
        hdr.set_data_shape((1, 2))
        assert_array_equal(hdr.get_zooms(), (4, 5))
        hdr.set_data_shape((1, 2, 3))
        assert_array_equal(hdr.get_zooms(), (4, 5, 1))
        assert_array_equal(np.diag(hdr.get_base_affine()), [-4, 5, 1, 1])
        hdr.set_zooms((1, 1, 1))
        assert_array_equal(np.diag(hdr.get_base_affine()), [-1, 1, 1, 1])

    def test_default_x_flip(self):
        hdr = self.header_class()
        hdr.default_x_flip = True
        hdr.set_data_shape((1, 2, 3))
        hdr.set_zooms((1, 1, 1))
        assert_array_equal(np.diag(hdr.get_base_affine()), [-1, 1, 1, 1])
        hdr.default_x_flip = False
        assert_array_equal(np.diag(hdr.get_base_affine()), [1, 1, 1, 1])

    def test_from_eg_file(self):
        fileobj = open(self.example_file, 'rb')
        hdr = self.header_class.from_fileobj(fileobj, check=False)
        assert hdr.endianness == '>'
        assert hdr['sizeof_hdr'] == self.sizeof_hdr

    def test_orientation(self):
        hdr = self.header_class()
        assert hdr.default_x_flip
        hdr.set_data_shape((3, 5, 7))
        hdr.set_zooms((4, 5, 6))
        aff = np.diag((-4, 5, 6, 1))
        aff[:3, 3] = np.array([1, 2, 3]) * np.array([-4, 5, 6]) * -1
        assert_array_equal(hdr.get_base_affine(), aff)
        hdr.default_x_flip = False
        assert not hdr.default_x_flip
        aff[0] *= -1
        assert_array_equal(hdr.get_base_affine(), aff)

    def test_str(self):
        super().test_str()
        hdr = self.header_class()
        s1 = str(hdr)
        rexp = re.compile('^datatype +: float32', re.MULTILINE)
        assert rexp.search(s1) is not None

    def test_from_header(self):
        klass = self.header_class
        empty = klass.from_header()
        assert klass() == empty
        empty = klass.from_header(None)
        assert klass() == empty
        hdr = klass()
        hdr.set_data_dtype(np.float64)
        hdr.set_data_shape((1, 2, 3))
        hdr.set_zooms((3.0, 2.0, 1.0))
        for check in (True, False):
            copy = klass.from_header(hdr, check=check)
            assert hdr == copy
            assert hdr is not copy

        class C:

            def get_data_dtype(self):
                return np.dtype('i2')

            def get_data_shape(self):
                return (5, 4, 3)

            def get_zooms(self):
                return (10.0, 9.0, 8.0)
        converted = klass.from_header(C())
        assert isinstance(converted, klass)
        assert converted.get_data_dtype() == np.dtype('i2')
        assert converted.get_data_shape() == (5, 4, 3)
        assert converted.get_zooms() == (10.0, 9.0, 8.0)

    def test_base_affine(self):
        klass = self.header_class
        hdr = klass()
        hdr.set_data_shape((3, 5, 7))
        hdr.set_zooms((3, 2, 1))
        assert hdr.default_x_flip
        assert_array_almost_equal(hdr.get_base_affine(), [[-3.0, 0.0, 0.0, 3.0], [0.0, 2.0, 0.0, -4.0], [0.0, 0.0, 1.0, -3.0], [0.0, 0.0, 0.0, 1.0]])
        hdr.set_data_shape((3, 5))
        assert_array_almost_equal(hdr.get_base_affine(), [[-3.0, 0.0, 0.0, 3.0], [0.0, 2.0, 0.0, -4.0], [0.0, 0.0, 1.0, -0.0], [0.0, 0.0, 0.0, 1.0]])
        hdr.set_data_shape((3, 5, 7))
        assert_array_almost_equal(hdr.get_base_affine(), [[-3.0, 0.0, 0.0, 3.0], [0.0, 2.0, 0.0, -4.0], [0.0, 0.0, 1.0, -3.0], [0.0, 0.0, 0.0, 1.0]])

    def test_scaling(self):
        hdr = self.header_class()
        assert hdr.default_x_flip
        shape = (1, 2, 3)
        hdr.set_data_shape(shape)
        hdr.set_data_dtype(np.float32)
        data = np.ones(shape, dtype=np.float64)
        S = BytesIO()
        hdr.data_to_fileobj(data, S)
        rdata = hdr.data_from_fileobj(S)
        assert_array_almost_equal(data, rdata)
        hdr.set_data_dtype(np.int32)
        if not hdr.has_data_slope:
            with pytest.raises(HeaderTypeError):
                hdr.data_to_fileobj(data, BytesIO())
        with np.errstate(invalid='ignore'):
            hdr.data_to_fileobj(data, S, rescale=False)
        rdata = hdr.data_from_fileobj(S)
        assert np.allclose(data, rdata)
        data_p5 = data + 0.5
        with np.errstate(invalid='ignore'):
            hdr.data_to_fileobj(data_p5, S, rescale=False)
        rdata = hdr.data_from_fileobj(S)
        assert not np.allclose(data_p5, rdata)

    def test_slope_inter(self):
        hdr = self.header_class()
        assert hdr.get_slope_inter() == (None, None)
        for slinter in ((None,), (None, None), (np.nan, np.nan), (np.nan, None), (None, np.nan), (1.0,), (1.0, None), (None, 0), (1.0, 0)):
            hdr.set_slope_inter(*slinter)
            assert hdr.get_slope_inter() == (None, None)
        with pytest.raises(HeaderTypeError):
            hdr.set_slope_inter(1.1)
        with pytest.raises(HeaderTypeError):
            hdr.set_slope_inter(1.0, 0.1)

    def test_from_analyze_map(self):
        klass = self.header_class

        class H1:
            pass
        with pytest.raises(AttributeError):
            klass.from_header(H1())

        class H2:

            def get_data_dtype(self):
                return np.dtype('u1')
        with pytest.raises(AttributeError):
            klass.from_header(H2())

        class H3(H2):

            def get_data_shape(self):
                return (2, 3, 4)
        with pytest.raises(AttributeError):
            klass.from_header(H3())

        class H4(H3):

            def get_zooms(self):
                return (4.0, 5.0, 6.0)
        exp_hdr = klass()
        exp_hdr.set_data_dtype(np.dtype('u1'))
        exp_hdr.set_data_shape((2, 3, 4))
        exp_hdr.set_zooms((4, 5, 6))
        assert klass.from_header(H4()) == exp_hdr

        class H5(H4):

            def as_analyze_map(self):
                return dict(cal_min=-100, cal_max=100)
        exp_hdr['cal_min'] = -100
        exp_hdr['cal_max'] = 100
        assert klass.from_header(H5()) == exp_hdr

        class H6(H5):

            def as_analyze_map(self):
                return dict(datatype=4, bitpix=32, cal_min=-100, cal_max=100)
        assert klass.from_header(H6()) == exp_hdr

        class H7(H5):

            def as_analyze_map(self):
                n_hdr = Nifti1Header()
                n_hdr.set_data_dtype(np.dtype('i2'))
                n_hdr['cal_min'] = -100
                n_hdr['cal_max'] = 100
                return n_hdr
        assert klass.from_header(H7()) == exp_hdr