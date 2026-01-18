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
def _check_array_caching(self, imaker, meth_name, caching):
    img = imaker()
    method = getattr(img, meth_name)
    get_data_func = method if caching is None else partial(method, caching=caching)
    assert isinstance(img.dataobj, np.ndarray)
    assert img.in_memory
    data = get_data_func()
    arr_dtype = img.dataobj.dtype
    dataobj_is_data = arr_dtype == np.float64 or method == img.get_data
    data[:] = 42
    get_result_changed = np.all(get_data_func() == 42)
    assert get_result_changed == (dataobj_is_data or caching != 'unchanged')
    if dataobj_is_data:
        assert data is img.dataobj
        assert_array_equal(np.asarray(img.dataobj), 42)
        img.uncache()
        assert_array_equal(get_data_func(), 42)
    else:
        assert not data is img.dataobj
        assert not np.all(np.asarray(img.dataobj) == 42)
        img.uncache()
        assert not np.all(get_data_func() == 42)
    img.uncache()
    assert img.in_memory
    if meth_name != 'get_fdata':
        return
    float_types = sctypes['float']
    if arr_dtype not in float_types:
        return
    for float_type in float_types:
        data = get_data_func(dtype=float_type)
        assert (data is img.dataobj) == (arr_dtype == float_type)