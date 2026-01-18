import bz2
import gzip
import types
import warnings
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import Nifti1Image, load, minc1
from ..deprecated import ModuleProxy
from ..deprecator import ExpiredDeprecationError
from ..externals.netcdf import netcdf_file
from ..minc1 import Minc1File, Minc1Image, MincHeader
from ..optpkg import optional_package
from ..testing import assert_data_similar, clear_and_catch_warnings, data_path
from ..tmpdirs import InTemporaryDirectory
from . import test_spatialimages as tsi
from .test_fileslice import slicer_samples
class TestMinc1File(_TestMincFile):

    def test_compressed(self):
        for tp in self.test_files:
            content = open(tp['fname'], 'rb').read()
            openers_exts = [(gzip.open, '.gz'), (bz2.BZ2File, '.bz2')]
            if HAVE_ZSTD:
                openers_exts += [(pyzstd.ZstdFile, '.zst')]
            with InTemporaryDirectory():
                for opener, ext in openers_exts:
                    fname = 'test.mnc' + ext
                    fobj = opener(fname, 'wb')
                    fobj.write(content)
                    fobj.close()
                    img = self.module.load(fname)
                    data = img.get_fdata()
                    assert_data_similar(data, tp)
                    del img