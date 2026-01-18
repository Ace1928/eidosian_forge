import unittest
import warnings
from io import BytesIO
from itertools import product
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from .. import ecat, minc1, minc2, parrec
from ..analyze import AnalyzeHeader
from ..arrayproxy import ArrayProxy, is_proxy
from ..casting import have_binary128, sctypes
from ..externals.netcdf import netcdf_file
from ..freesurfer.mghformat import MGHHeader
from ..nifti1 import Nifti1Header
from ..optpkg import optional_package
from ..spm2analyze import Spm2AnalyzeHeader
from ..spm99analyze import Spm99AnalyzeHeader
from ..testing import assert_dt_equal, clear_and_catch_warnings
from ..testing import data_path as DATA_PATH
from ..tmpdirs import InTemporaryDirectory
from ..volumeutils import apply_read_scaling
from .test_api_validators import ValidateAPI
from .test_parrec import EG_REC, VARY_REC
class _TestProxyAPI(ValidateAPI):
    """Base class for testing proxy APIs

    Assumes that real classes will provide an `obj_params` method which is a
    generator returning 2 tuples of (<proxy_maker>, <param_dict>).
    <proxy_maker> is a function returning a 3 tuple of (<proxy>, <fileobj>,
    <header>).  <param_dict> is a dictionary containing at least keys
    ``arr_out`` (expected output array from proxy), ``dtype_out`` (expected
    output dtype for array) and ``shape`` (shape of array).

    The <header> above should support at least "get_data_dtype",
    "set_data_dtype", "get_data_shape", "set_data_shape"
    """
    settable_offset = False

    def validate_shape(self, pmaker, params):
        prox, fio, hdr = pmaker()
        assert_array_equal(prox.shape, params['shape'])
        with pytest.raises(AttributeError):
            prox.shape = params['shape']

    def validate_ndim(self, pmaker, params):
        prox, fio, hdr = pmaker()
        assert prox.ndim == len(params['shape'])
        with pytest.raises(AttributeError):
            prox.ndim = len(params['shape'])

    def validate_is_proxy(self, pmaker, params):
        prox, fio, hdr = pmaker()
        assert prox.is_proxy
        assert is_proxy(prox)
        assert not is_proxy(np.arange(10))
        with pytest.raises(AttributeError):
            prox.is_proxy = False

    def validate_asarray(self, pmaker, params):
        prox, fio, hdr = pmaker()
        out = np.asarray(prox)
        assert_array_equal(out, params['arr_out'])
        assert_dt_equal(out.dtype, params['dtype_out'])
        assert out.shape == params['shape']

    def validate_array_interface_with_dtype(self, pmaker, params):
        prox, fio, hdr = pmaker()
        orig = np.array(prox, dtype=None)
        assert_array_equal(orig, params['arr_out'])
        assert_dt_equal(orig.dtype, params['dtype_out'])
        context = None
        if np.issubdtype(orig.dtype, np.complexfloating):
            context = clear_and_catch_warnings()
            context.__enter__()
            warnings.simplefilter('ignore', ComplexWarning)
        for dtype in sctypes['float'] + sctypes['int'] + sctypes['uint']:
            direct = dtype(prox)
            rtol = 0.001 if dtype == np.float16 else 1e-05
            assert_allclose(direct, orig.astype(dtype), rtol=rtol, atol=1e-08)
            assert_dt_equal(direct.dtype, np.dtype(dtype))
            assert direct.shape == params['shape']
            for arrmethod in (np.array, np.asarray, np.asanyarray):
                out = arrmethod(prox, dtype=dtype)
                assert_array_equal(out, direct)
                assert_dt_equal(out.dtype, np.dtype(dtype))
                assert out.shape == params['shape']
        if context is not None:
            context.__exit__()

    def validate_header_isolated(self, pmaker, params):
        prox, fio, hdr = pmaker()
        assert_array_equal(prox, params['arr_out'])
        if hdr.get_data_dtype() == np.uint8:
            hdr.set_data_dtype(np.int16)
        else:
            hdr.set_data_dtype(np.uint8)
        hdr.set_data_shape(np.array(hdr.get_data_shape()) + 1)
        if self.settable_offset:
            hdr.set_data_offset(32)
        assert_array_equal(prox, params['arr_out'])

    def validate_fileobj_isolated(self, pmaker, params):
        prox, fio, hdr = pmaker()
        if isinstance(fio, str):
            return
        assert_array_equal(prox, params['arr_out'])
        fio.read()
        assert_array_equal(prox, params['arr_out'])

    def validate_proxy_slicing(self, pmaker, params):
        arr = params['arr_out']
        shape = arr.shape
        prox, fio, hdr = pmaker()
        for sliceobj in _some_slicers(shape):
            assert_array_equal(arr[sliceobj], prox[sliceobj])