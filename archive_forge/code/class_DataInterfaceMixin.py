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
class DataInterfaceMixin(GetSetDtypeMixin):
    """Test dataobj interface for images with array backing

    Use this mixin if your image has a ``dataobj`` property that contains an
    array or an array-like thing.
    """
    meth_names = ('get_fdata',)

    def validate_data_interface(self, imaker, params):
        img = imaker()
        assert img.shape == img.dataobj.shape
        assert img.ndim == len(img.shape)
        assert_data_similar(img.dataobj, params)
        for meth_name in self.meth_names:
            if params['is_proxy']:
                self._check_proxy_interface(imaker, meth_name)
            else:
                self._check_array_interface(imaker, meth_name)
            method = getattr(img, meth_name)
            assert img.shape == method().shape
            assert img.ndim == method().ndim
            with pytest.raises(ValueError):
                method(caching='something')
        fake_data = np.zeros(img.shape, dtype=img.get_data_dtype())
        with pytest.raises(AttributeError):
            img.dataobj = fake_data
        with pytest.raises(AttributeError):
            img.in_memory = False

    def _check_proxy_interface(self, imaker, meth_name):
        img = imaker()
        assert is_proxy(img.dataobj)
        assert not isinstance(img.dataobj, np.ndarray)
        proxy_data = np.asarray(img.dataobj)
        proxy_copy = proxy_data.copy()
        assert not img.in_memory
        method = getattr(img, meth_name)
        data = method(caching='unchanged')
        assert not img.in_memory
        data = method()
        assert img.in_memory
        assert not proxy_data is data
        assert_array_equal(proxy_data, data)
        data_again = method(caching='unchanged')
        assert data is data_again
        data_yet_again = method(caching='fill')
        assert data is data_yet_again
        data[:] = 42
        assert_array_equal(proxy_data, proxy_copy)
        assert_array_equal(np.asarray(img.dataobj), proxy_copy)
        assert_array_equal(method(), 42)
        img.uncache()
        assert not img.in_memory
        assert_array_equal(method(), proxy_copy)
        img = imaker()
        method = getattr(img, meth_name)
        assert not img.in_memory
        data = method(caching='fill')
        assert img.in_memory
        data_again = method()
        assert data is data_again
        img.uncache()
        fdata = img.get_fdata()
        assert fdata.dtype == np.float64
        fdata[:] = 42
        fdata_back = img.get_fdata()
        assert_array_equal(fdata_back, 42)
        assert fdata_back.dtype == np.float64
        fdata_new_dt = img.get_fdata(caching='unchanged', dtype='f4')
        assert_allclose(fdata_new_dt, proxy_data.astype('f4'), rtol=1e-05, atol=1e-08)
        assert fdata_new_dt.dtype == np.float32
        assert_array_equal(img.get_fdata(), 42)
        fdata_new_dt[:] = 43
        fdata_new_dt = img.get_fdata(caching='unchanged', dtype='f4')
        assert_allclose(fdata_new_dt, proxy_data.astype('f4'), rtol=1e-05, atol=1e-08)
        fdata_new_dt = img.get_fdata(caching='fill', dtype='f4')
        assert_allclose(fdata_new_dt, proxy_data.astype('f4'), rtol=1e-05, atol=1e-08)
        fdata_new_dt[:] = 43
        assert_array_equal(img.get_fdata(dtype='f4'), 43)
        assert_array_equal(img.get_fdata(), proxy_data)

    def _check_array_interface(self, imaker, meth_name):
        for caching in (None, 'fill', 'unchanged'):
            self._check_array_caching(imaker, meth_name, caching)

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

    def validate_shape(self, imaker, params):
        img = imaker()
        assert img.shape == params['shape']
        if 'data' in params:
            assert img.shape == params['data'].shape
        with pytest.raises(AttributeError):
            img.shape = np.eye(4)

    def validate_ndim(self, imaker, params):
        img = imaker()
        assert img.ndim == len(params['shape'])
        if 'data' in params:
            assert img.ndim == params['data'].ndim
        with pytest.raises(AttributeError):
            img.ndim = 5

    def validate_mmap_parameter(self, imaker, params):
        img = imaker()
        fname = img.get_filename()
        with InTemporaryDirectory():
            if fname is None:
                if not img.rw or not img.valid_exts:
                    return
                fname = 'image' + img.valid_exts[0]
                img.to_filename(fname)
            rt_img = img.__class__.from_filename(fname, mmap=True)
            assert_almost_equal(img.get_fdata(), rt_img.get_fdata())
            rt_img = img.__class__.from_filename(fname, mmap=False)
            assert_almost_equal(img.get_fdata(), rt_img.get_fdata())
            rt_img = img.__class__.from_filename(fname, mmap='c')
            assert_almost_equal(img.get_fdata(), rt_img.get_fdata())
            rt_img = img.__class__.from_filename(fname, mmap='r')
            assert_almost_equal(img.get_fdata(), rt_img.get_fdata())
            with pytest.raises(ValueError):
                img.__class__.from_filename(fname, mmap='r+')
            with pytest.raises(ValueError):
                img.__class__.from_filename(fname, mmap='invalid')
            del rt_img