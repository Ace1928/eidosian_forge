import io
import pathlib
import sys
import warnings
from functools import partial
from itertools import product
import numpy as np
from ..optpkg import optional_package
import unittest
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal
from nibabel.arraywriters import WriterError
from nibabel.testing import (
from .. import (
from ..casting import sctypes
from ..spatialimages import SpatialImage
from ..tmpdirs import InTemporaryDirectory
from .test_api_validators import ValidateAPI
from .test_brikhead import EXAMPLE_IMAGES as AFNI_EXAMPLE_IMAGES
from .test_minc1 import EXAMPLE_IMAGES as MINC1_EXAMPLE_IMAGES
from .test_minc2 import EXAMPLE_IMAGES as MINC2_EXAMPLE_IMAGES
from .test_parrec import EXAMPLE_IMAGES as PARREC_EXAMPLE_IMAGES
class HeaderShapeMixin:
    """Tests that header shape can be set and got

    Add this one of your header supports ``get_data_shape`` and
    ``set_data_shape``.
    """

    def validate_header_shape(self, imaker, params):
        img = imaker()
        hdr = img.header
        shape = hdr.get_data_shape()
        new_shape = (shape[0] + 1,) + shape[1:]
        hdr.set_data_shape(new_shape)
        assert img.header is hdr
        assert img.header.get_data_shape() == new_shape