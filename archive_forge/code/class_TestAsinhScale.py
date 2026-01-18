import copy
import matplotlib.pyplot as plt
from matplotlib.scale import (
import matplotlib.scale as mscale
from matplotlib.ticker import AsinhLocator, LogFormatterSciNotation
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import numpy as np
from numpy.testing import assert_allclose
import io
import pytest
class TestAsinhScale:

    def test_transforms(self):
        a0 = 17.0
        a = np.linspace(-50, 50, 100)
        forward = AsinhTransform(a0)
        inverse = forward.inverted()
        invinv = inverse.inverted()
        a_forward = forward.transform_non_affine(a)
        a_inverted = inverse.transform_non_affine(a_forward)
        assert_allclose(a_inverted, a)
        a_invinv = invinv.transform_non_affine(a)
        assert_allclose(a_invinv, a0 * np.arcsinh(a / a0))

    def test_init(self):
        fig, ax = plt.subplots()
        s = AsinhScale(axis=None, linear_width=23.0)
        assert s.linear_width == 23
        assert s._base == 10
        assert s._subs == (2, 5)
        tx = s.get_transform()
        assert isinstance(tx, AsinhTransform)
        assert tx.linear_width == s.linear_width

    def test_base_init(self):
        fig, ax = plt.subplots()
        s3 = AsinhScale(axis=None, base=3)
        assert s3._base == 3
        assert s3._subs == (2,)
        s7 = AsinhScale(axis=None, base=7, subs=(2, 4))
        assert s7._base == 7
        assert s7._subs == (2, 4)

    def test_fmtloc(self):

        class DummyAxis:

            def __init__(self):
                self.fields = {}

            def set(self, **kwargs):
                self.fields.update(**kwargs)

            def set_major_formatter(self, f):
                self.fields['major_formatter'] = f
        ax0 = DummyAxis()
        s0 = AsinhScale(axis=ax0, base=0)
        s0.set_default_locators_and_formatters(ax0)
        assert isinstance(ax0.fields['major_locator'], AsinhLocator)
        assert isinstance(ax0.fields['major_formatter'], str)
        ax5 = DummyAxis()
        s7 = AsinhScale(axis=ax5, base=5)
        s7.set_default_locators_and_formatters(ax5)
        assert isinstance(ax5.fields['major_locator'], AsinhLocator)
        assert isinstance(ax5.fields['major_formatter'], LogFormatterSciNotation)

    def test_bad_scale(self):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError):
            AsinhScale(axis=None, linear_width=0)
        with pytest.raises(ValueError):
            AsinhScale(axis=None, linear_width=-1)
        s0 = AsinhScale(axis=None)
        s1 = AsinhScale(axis=None, linear_width=3.0)