import contextlib
import gzip
import pickle
from io import BytesIO
from unittest import mock
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from packaging.version import Version
from .. import __version__
from ..arrayproxy import ArrayProxy, get_obj_dtype, is_proxy, reshape_dataobj
from ..deprecator import ExpiredDeprecationError
from ..nifti1 import Nifti1Header, Nifti1Image
from ..openers import ImageOpener
from ..testing import memmap_after_ufunc
from ..tmpdirs import InTemporaryDirectory
from .test_fileslice import slicer_samples
from .test_openers import patch_indexed_gzip
class FunkyHeader:

    def __init__(self, shape):
        self.shape = shape

    def get_data_shape(self):
        return self.shape[:]

    def get_data_dtype(self):
        return np.int32

    def get_data_offset(self):
        return 16

    def get_slope_inter(self):
        return (1.0, 0.0)

    def copy(self):
        return FunkyHeader(self.shape)