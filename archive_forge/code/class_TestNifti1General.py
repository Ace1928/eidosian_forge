import os
import struct
import unittest
import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from nibabel import nifti1 as nifti1
from nibabel.affines import from_matvec
from nibabel.casting import have_binary128, type_info
from nibabel.eulerangles import euler2mat
from nibabel.nifti1 import (
from nibabel.optpkg import optional_package
from nibabel.pkg_info import cmp_pkg_version
from nibabel.spatialimages import HeaderDataError
from nibabel.tmpdirs import InTemporaryDirectory
from ..freesurfer import load as mghload
from ..orientations import aff2axcodes
from ..testing import (
from . import test_analyze as tana
from . import test_spm99analyze as tspm
from .nibabel_data import get_nibabel_data, needs_nibabel_data
from .test_arraywriters import IUINT_TYPES, rt_err_estimate
from .test_orientations import ALL_ORNTS
class TestNifti1General:
    """Test class to test nifti1 in general

    Tests here which mix the pair and the single type, and that should only be
    run once (not for each type) because they are slow
    """
    single_class = Nifti1Image
    pair_class = Nifti1Pair
    module = nifti1
    example_file = image_file

    def test_loadsave_cycle(self):
        nim = self.module.load(self.example_file)
        hdr = nim.header
        exts_container = hdr.extensions
        assert len(exts_container) > 0
        lnim = bytesio_round_trip(nim)
        hdr = lnim.header
        lexts_container = hdr.extensions
        assert exts_container == lexts_container
        data = np.ones((2, 3, 4, 5), dtype='int16')
        img = self.single_class(data, np.eye(4))
        hdr = img.header
        assert hdr.get_data_dtype() == np.int16
        assert_array_equal(hdr.get_slope_inter(), (None, None))
        hdr.set_slope_inter(2, 8)
        assert hdr.get_slope_inter() == (2, 8)
        wnim = self.single_class(data, np.eye(4), header=hdr)
        assert wnim.get_data_dtype() == np.int16
        assert wnim.header.get_slope_inter() == (None, None)
        wnim.header.set_slope_inter(2, 8)
        assert wnim.header.get_slope_inter() == (2, 8)
        lnim = bytesio_round_trip(wnim)
        assert lnim.get_data_dtype() == np.int16
        assert_array_equal(lnim.get_fdata(), data * 2.0 + 8.0)
        assert lnim.header.get_slope_inter() == (None, None)
        assert (lnim.dataobj.slope, lnim.dataobj.inter) == (2, 8)

    def test_load(self):
        arr = np.arange(24, dtype='f4').reshape((2, 3, 4))
        aff = np.diag([2, 3, 4, 1])
        simg = self.single_class(arr, aff)
        pimg = self.pair_class(arr, aff)
        save = self.module.save
        load = self.module.load
        with InTemporaryDirectory():
            for img in (simg, pimg):
                save(img, 'test.nii')
                assert_array_equal(arr, load('test.nii').get_fdata())
                save(simg, 'test.img')
                assert_array_equal(arr, load('test.img').get_fdata())
                save(simg, 'test.hdr')
                assert_array_equal(arr, load('test.hdr').get_fdata())

    def test_float_int_min_max(self):
        aff = np.eye(4)
        for in_dt in (np.float32, np.float64):
            finf = type_info(in_dt)
            arr = np.array([finf['min'], finf['max']], dtype=in_dt)
            for out_dt in IUINT_TYPES:
                img = self.single_class(arr, aff)
                img_back = bytesio_round_trip(img)
                arr_back_sc = img_back.get_fdata()
                assert np.allclose(arr, arr_back_sc)

    def test_float_int_spread(self):
        powers = np.arange(-10, 10, 0.5)
        arr = np.concatenate((-10 ** powers, 10 ** powers))
        aff = np.eye(4)
        for in_dt in (np.float32, np.float64):
            arr_t = arr.astype(in_dt)
            for out_dt in IUINT_TYPES:
                img = self.single_class(arr_t, aff)
                img_back = bytesio_round_trip(img)
                arr_back_sc = img_back.get_fdata()
                slope, inter = img_back.header.get_slope_inter()
                max_miss = rt_err_estimate(arr_t, arr_back_sc.dtype, slope, inter)
                diff = np.abs(arr_t - arr_back_sc)
                rdiff = diff / np.abs(arr_t)
                assert np.all((diff <= max_miss) | (rdiff <= 1e-05))

    def test_rt_bias(self):
        rng = np.random.RandomState(20111214)
        mu, std, count = (100, 10, 100)
        arr = rng.normal(mu, std, size=(count,))
        eps = np.finfo(np.float32).eps
        aff = np.eye(4)
        for in_dt in (np.float32, np.float64):
            arr_t = arr.astype(in_dt)
            for out_dt in IUINT_TYPES:
                img = self.single_class(arr_t, aff)
                img_back = bytesio_round_trip(img)
                arr_back_sc = img_back.get_fdata()
                slope, inter = img_back.header.get_slope_inter()
                bias = np.mean(arr_t - arr_back_sc)
                max_miss = rt_err_estimate(arr_t, arr_back_sc.dtype, slope, inter)
                bias_thresh = np.max([max_miss / np.sqrt(count), eps])
                assert np.abs(bias) < bias_thresh

    def test_reoriented_dim_info(self):
        arr = np.arange(24, dtype='f4').reshape((2, 3, 4))
        aff = np.diag([2, 3, 4, 1])
        simg = self.single_class(arr, aff)
        for freq, phas, slic in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 0, 1), (None, None, None), (0, 2, None), (0, None, None), (None, 2, 1), (None, None, 1)):
            simg.header.set_dim_info(freq, phas, slic)
            fdir = 'RAS'[freq] if freq is not None else None
            pdir = 'RAS'[phas] if phas is not None else None
            sdir = 'RAS'[slic] if slic is not None else None
            for ornt in ALL_ORNTS:
                rimg = simg.as_reoriented(np.array(ornt))
                axcode = aff2axcodes(rimg.affine)
                dirs = ''.join(axcode).replace('P', 'A').replace('I', 'S').replace('L', 'R')
                new_freq, new_phas, new_slic = rimg.header.get_dim_info()
                new_fdir = dirs[new_freq] if new_freq is not None else None
                new_pdir = dirs[new_phas] if new_phas is not None else None
                new_sdir = dirs[new_slic] if new_slic is not None else None
                assert (new_fdir, new_pdir, new_sdir) == (fdir, pdir, sdir)