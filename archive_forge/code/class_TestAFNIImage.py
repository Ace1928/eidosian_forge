from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import Nifti1Image, brikhead, load
from ..testing import assert_data_similar, data_path
from .test_fileslice import slicer_samples
class TestAFNIImage:
    module = brikhead
    test_files = EXAMPLE_IMAGES

    def test_brikheadfile(self):
        for tp in self.test_files:
            brik = self.module.load(tp['fname'])
            assert brik.get_data_dtype().type == tp['dtype']
            assert brik.shape == tp['shape']
            assert brik.header.get_zooms() == tp['zooms']
            assert_array_equal(brik.affine, tp['affine'])
            assert brik.header.get_space() == tp['space']
            data = brik.get_fdata()
            assert data.shape == tp['shape']
            assert_array_equal(brik.dataobj.scaling, tp['scaling'])
            assert brik.header.get_volume_labels() == tp['labels']

    def test_load(self):
        for tp in self.test_files:
            img = self.module.load(tp['head'])
            data = img.get_fdata()
            assert data.shape == tp['shape']
            assert_data_similar(data, tp)
            ni_img = Nifti1Image.from_image(img)
            assert_array_equal(ni_img.affine, tp['affine'])
            assert_array_equal(ni_img.get_fdata(), data)

    def test_array_proxy_slicing(self):
        for tp in self.test_files:
            img = self.module.load(tp['fname'])
            arr = img.get_fdata()
            prox = img.dataobj
            assert prox.is_proxy
            for sliceobj in slicer_samples(img.shape):
                assert_array_equal(arr[sliceobj], prox[sliceobj])