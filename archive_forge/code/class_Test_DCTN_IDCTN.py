from os.path import join, dirname
from typing import Callable, Union
import numpy as np
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft.realtransforms import (
class Test_DCTN_IDCTN:
    dec = 14
    dct_type = [1, 2, 3, 4]
    norms = [None, 'backward', 'ortho', 'forward']
    rstate = np.random.RandomState(1234)
    shape = (32, 16)
    data = rstate.randn(*shape)

    @pytest.mark.parametrize('fforward,finverse', [(dctn, idctn), (dstn, idstn)])
    @pytest.mark.parametrize('axes', [None, 1, (1,), [1], 0, (0,), [0], (0, 1), [0, 1], (-2, -1), [-2, -1]])
    @pytest.mark.parametrize('dct_type', dct_type)
    @pytest.mark.parametrize('norm', ['ortho'])
    def test_axes_round_trip(self, fforward, finverse, axes, dct_type, norm):
        tmp = fforward(self.data, type=dct_type, axes=axes, norm=norm)
        tmp = finverse(tmp, type=dct_type, axes=axes, norm=norm)
        assert_array_almost_equal(self.data, tmp, decimal=12)

    @pytest.mark.parametrize('funcn,func', [(dctn, dct), (dstn, dst)])
    @pytest.mark.parametrize('dct_type', dct_type)
    @pytest.mark.parametrize('norm', norms)
    def test_dctn_vs_2d_reference(self, funcn, func, dct_type, norm):
        y1 = funcn(self.data, type=dct_type, axes=None, norm=norm)
        y2 = ref_2d(func, self.data, type=dct_type, norm=norm)
        assert_array_almost_equal(y1, y2, decimal=11)

    @pytest.mark.parametrize('funcn,func', [(idctn, idct), (idstn, idst)])
    @pytest.mark.parametrize('dct_type', dct_type)
    @pytest.mark.parametrize('norm', norms)
    def test_idctn_vs_2d_reference(self, funcn, func, dct_type, norm):
        fdata = dctn(self.data, type=dct_type, norm=norm)
        y1 = funcn(fdata, type=dct_type, norm=norm)
        y2 = ref_2d(func, fdata, type=dct_type, norm=norm)
        assert_array_almost_equal(y1, y2, decimal=11)

    @pytest.mark.parametrize('fforward,finverse', [(dctn, idctn), (dstn, idstn)])
    def test_axes_and_shape(self, fforward, finverse):
        with assert_raises(ValueError, match='when given, axes and shape arguments have to be of the same length'):
            fforward(self.data, s=self.data.shape[0], axes=(0, 1))
        with assert_raises(ValueError, match='when given, axes and shape arguments have to be of the same length'):
            fforward(self.data, s=self.data.shape, axes=0)

    @pytest.mark.parametrize('fforward', [dctn, dstn])
    def test_shape(self, fforward):
        tmp = fforward(self.data, s=(128, 128), axes=None)
        assert_equal(tmp.shape, (128, 128))

    @pytest.mark.parametrize('fforward,finverse', [(dctn, idctn), (dstn, idstn)])
    @pytest.mark.parametrize('axes', [1, (1,), [1], 0, (0,), [0]])
    def test_shape_is_none_with_axes(self, fforward, finverse, axes):
        tmp = fforward(self.data, s=None, axes=axes, norm='ortho')
        tmp = finverse(tmp, s=None, axes=axes, norm='ortho')
        assert_array_almost_equal(self.data, tmp, decimal=self.dec)