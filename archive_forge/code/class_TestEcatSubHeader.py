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
class TestEcatSubHeader(TestCase):
    header_class = EcatHeader
    subhdr_class = EcatSubHeader
    example_file = ecat_file
    fid = open(example_file, 'rb')
    hdr = header_class.from_fileobj(fid)
    mlist = read_mlist(fid, hdr.endianness)
    subhdr = subhdr_class(hdr, mlist, fid)

    def test_subheader_size(self):
        assert self.subhdr_class._subhdrdtype.itemsize == 510

    def test_subheader(self):
        assert self.subhdr.get_shape() == (10, 10, 3)
        assert self.subhdr.get_nframes() == 1
        assert self.subhdr.get_nframes() == len(self.subhdr.subheaders)
        assert self.subhdr._check_affines() is True
        assert_array_almost_equal(np.diag(self.subhdr.get_frame_affine()), np.array([2.20241979, 2.20241979, 3.125, 1.0]))
        assert self.subhdr.get_zooms()[0] == 2.20241978764534
        assert self.subhdr.get_zooms()[2] == 3.125
        assert self.subhdr._get_data_dtype(0) == np.int16
        assert self.subhdr._get_frame_offset() == 1536
        dat = self.subhdr.raw_data_from_fileobj()
        assert dat.shape == self.subhdr.get_shape()
        assert self.subhdr.subheaders[0]['scale_factor'].item() == 1.0
        ecat_calib_factor = self.hdr['ecat_calibration_factor']
        assert ecat_calib_factor == 25007614.0