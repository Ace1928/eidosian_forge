import itertools
import sys
import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nibabel.tmpdirs import InTemporaryDirectory
from ... import load
from ...fileholders import FileHolder
from ...nifti1 import data_type_codes
from ...testing import get_test_data
from .. import (
from .test_parse_gifti_fast import (
def _check_gifti(gio):
    vertices = gio.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
    faces = gio.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
    assert_array_equal(vertices, exp_verts)
    assert_array_equal(faces, exp_faces)