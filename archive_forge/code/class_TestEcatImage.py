import os
import warnings
from pathlib import Path
from unittest import TestCase
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..ecat import (
from ..openers import Opener
from ..testing import data_path, suppress_warnings
from ..tmpdirs import InTemporaryDirectory
from . import test_wrapstruct as tws
from .test_fileslice import slicer_samples
class TestEcatImage(TestCase):
    image_class = EcatImage
    example_file = ecat_file
    img = image_class.load(example_file)

    def test_file(self):
        assert Path(self.img.file_map['header'].filename) == Path(self.example_file)
        assert Path(self.img.file_map['image'].filename) == Path(self.example_file)

    def test_save(self):
        tmp_file = 'tinypet_tmp.v'
        with InTemporaryDirectory():
            self.img.to_filename(tmp_file)
            other = self.image_class.load(tmp_file)
            assert_array_equal(self.img.get_fdata(), other.get_fdata())
            del other

    def test_data(self):
        dat = self.img.get_fdata()
        assert dat.shape == self.img.shape
        frame = self.img.get_frame(0)
        assert_array_equal(frame, dat[:, :, :, 0])

    def test_array_proxy(self):
        dat = self.img.get_fdata()
        img = self.image_class.load(self.example_file)
        data_prox = img.dataobj
        data2 = np.array(data_prox)
        assert_array_equal(data2, dat)
        data3 = np.array(data_prox)
        assert_array_equal(data3, dat)

    def test_array_proxy_slicing(self):
        arr = self.img.get_fdata()
        prox = self.img.dataobj
        assert prox.is_proxy
        for sliceobj in slicer_samples(self.img.shape):
            assert_array_equal(arr[sliceobj], prox[sliceobj])

    def test_isolation(self):
        img_klass = self.image_class
        arr, aff, hdr, sub_hdr, mlist = (self.img.get_fdata(), self.img.affine, self.img.header, self.img.get_subheaders(), self.img.get_mlist())
        img = img_klass(arr, aff, hdr, sub_hdr, mlist)
        assert_array_equal(img.affine, aff)
        aff[0, 0] = 99
        assert not np.all(img.affine == aff)

    def test_float_affine(self):
        img_klass = self.image_class
        arr, aff, hdr, sub_hdr, mlist = (self.img.get_fdata(), self.img.affine, self.img.header, self.img.get_subheaders(), self.img.get_mlist())
        img = img_klass(arr, aff.astype(np.float32), hdr, sub_hdr, mlist)
        assert img.affine.dtype == np.dtype(np.float64)
        img = img_klass(arr, aff.astype(np.int16), hdr, sub_hdr, mlist)
        assert img.affine.dtype == np.dtype(np.float64)

    def test_data_regression(self):
        vals = dict(max=248750736458.0, min=1125342630.0, mean=117907565661.46666)
        data = self.img.get_fdata()
        assert data.max() == vals['max']
        assert data.min() == vals['min']
        assert_array_almost_equal(data.mean(), vals['mean'])

    def test_mlist_regression(self):
        assert_array_equal(self.img.get_mlist(), [[16842758, 3, 3011, 1]])