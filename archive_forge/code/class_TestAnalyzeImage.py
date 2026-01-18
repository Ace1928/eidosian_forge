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
class TestAnalyzeImage(tsi.TestSpatialImage, tsi.MmapImageMixin):
    image_class = AnalyzeImage
    can_save = True
    supported_np_types = TestAnalyzeHeader.supported_np_types

    def test_supported_types(self):
        img = self.image_class(np.zeros((2, 3, 4)), np.eye(4))
        assert self.supported_np_types == supported_np_types(img)

    def test_default_header(self):
        arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        img = self.image_class(arr, None)
        hdr = self.image_class.header_class()
        hdr.set_data_shape(arr.shape)
        hdr.set_data_dtype(arr.dtype)
        hdr.set_data_offset(0)
        hdr.set_slope_inter(np.nan, np.nan)
        assert img.header == hdr

    def test_data_hdr_cache(self):
        IC = self.image_class
        fm = IC.make_file_map()
        for key, value in fm.items():
            fm[key].fileobj = BytesIO()
        shape = (2, 3, 4)
        data = np.arange(24, dtype=np.int8).reshape(shape)
        affine = np.eye(4)
        hdr = IC.header_class()
        hdr.set_data_dtype(np.int16)
        img = IC(data, affine, hdr)
        img.to_file_map(fm)
        img2 = IC.from_file_map(fm)
        assert img2.shape == shape
        assert img2.get_data_dtype().type == np.int16
        hdr = img2.header
        hdr.set_data_shape((3, 2, 2))
        assert hdr.get_data_shape() == (3, 2, 2)
        hdr.set_data_dtype(np.uint8)
        assert hdr.get_data_dtype() == np.dtype(np.uint8)
        assert_array_equal(img2.get_fdata(), data)
        assert_array_equal(np.asanyarray(img2.dataobj), data)

    def test_affine_44(self):
        IC = self.image_class
        shape = (2, 3, 4)
        data = np.arange(24, dtype=np.int16).reshape(shape)
        affine = np.diag([2, 3, 4, 1])
        img = IC(data, affine)
        assert_array_equal(affine, img.affine)
        img = IC(data, affine.tolist())
        assert_array_equal(affine, img.affine)
        with pytest.raises(ValueError):
            IC(data, np.diag([2, 3, 4]))

    def test_dtype_init_arg(self):
        img_klass = self.image_class
        arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        aff = np.eye(4)
        for dtype in self.supported_np_types:
            img = img_klass(arr, aff, dtype=dtype)
            assert img.get_data_dtype() == dtype
        hdr = img.header
        for dtype in self.supported_np_types:
            img = img_klass(arr, aff, hdr, dtype=dtype)
            assert img.get_data_dtype() == dtype

    def test_offset_to_zero(self):
        img_klass = self.image_class
        arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        aff = np.eye(4)
        img = img_klass(arr, aff)
        assert img.header.get_data_offset() == 0
        bytes_map = bytesio_filemap(img_klass)
        img.to_file_map(bytes_map)
        assert img.header.get_data_offset() == 0
        big_off = 1024
        img.header.set_data_offset(big_off)
        assert img.header.get_data_offset() == big_off
        img_rt = bytesio_round_trip(img)
        assert img_rt.dataobj.offset == big_off
        assert img_rt.header.get_data_offset() == 0
        img.header.set_data_offset(big_off)
        img_again = img_klass(arr, aff, img.header)
        assert img_again.header.get_data_offset() == 0

    def test_big_offset_exts(self):
        img_klass = self.image_class
        arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        aff = np.eye(4)
        img_ext = img_klass.files_types[0][1]
        compressed_exts = ['', '.gz', '.bz2']
        if HAVE_ZSTD:
            compressed_exts += ['.zst']
        with InTemporaryDirectory():
            for offset in (0, 2048):
                for compressed_ext in compressed_exts:
                    img = img_klass(arr, aff)
                    img.header.set_data_offset(offset)
                    fname = 'test' + img_ext + compressed_ext
                    img.to_filename(fname)
                    img_back = img_klass.from_filename(fname)
                    assert_array_equal(arr, img_back.dataobj)
            del img, img_back

    def test_header_updating(self):
        img_klass = self.image_class
        img = img_klass(np.zeros((2, 3, 4)), None)
        hdr = img.header
        hdr.set_zooms((4, 5, 6))
        for key, value in img.file_map.items():
            value.fileobj = BytesIO()
        img.to_file_map()
        hdr_back = img.from_file_map(img.file_map).header
        assert_array_equal(hdr_back.get_zooms(), (4, 5, 6))
        img = img_klass(np.zeros((2, 3, 4)), np.diag([2, 3, 4, 1]), hdr)
        hdr = img.header
        assert_array_equal(hdr.get_zooms(), (2, 3, 4))
        img.affine[0, 0] = 9
        for key, value in img.file_map.items():
            value.fileobj = BytesIO()
        img.to_file_map()
        hdr_back = img.from_file_map(img.file_map).header
        assert_array_equal(hdr.get_zooms(), (9, 3, 4))
        data = img.get_fdata()
        data.shape = (3, 2, 4)
        img.to_file_map()
        img_back = img.from_file_map(img.file_map)
        assert_array_equal(img_back.shape, (3, 2, 4))

    def test_pickle(self):
        img_klass = self.image_class
        img = img_klass(np.zeros((2, 3, 4)), None)
        img_str = pickle.dumps(img)
        img2 = pickle.loads(img_str)
        assert_array_equal(img.get_fdata(), img2.get_fdata())
        assert img.header == img2.header
        for key, value in img.file_map.items():
            value.fileobj = BytesIO()
        img.to_file_map()
        img_prox = img.from_file_map(img.file_map)
        img_str = pickle.dumps(img_prox)
        img2_prox = pickle.loads(img_str)
        assert_array_equal(img.get_fdata(), img2_prox.get_fdata())

    def test_no_finite_values(self):
        data = np.zeros((2, 3, 4))
        data[:, 0] = np.nan
        data[:, 1] = np.inf
        data[:, 2] = -np.inf
        img = self.image_class(data, None)
        img.set_data_dtype(np.int16)
        assert img.get_data_dtype() == np.dtype(np.int16)
        fm = bytesio_filemap(img)
        if not img.header.has_data_slope:
            with pytest.raises(WriterError):
                img.to_file_map(fm)
            return
        img.to_file_map(fm)
        img_back = self.image_class.from_file_map(fm)
        assert_array_equal(img_back.dataobj, 0)

    def test_dtype_to_filename_arg(self):
        img_klass = self.image_class
        arr = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        aff = np.eye(4)
        img = img_klass(arr, aff)
        fname = 'test' + img_klass.files_types[0][1]
        with InTemporaryDirectory():
            for dtype in self.supported_np_types:
                img.to_filename(fname, dtype=dtype)
                new_img = img_klass.from_filename(fname)
                assert new_img.get_data_dtype() == dtype
                assert img.get_data_dtype() == np.int16