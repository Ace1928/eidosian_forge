from __future__ import annotations
import contextlib
import inspect
import math
from collections.abc import Hashable
from copy import copy
from datetime import date, datetime, timedelta
from typing import Any, Callable, Literal
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xarray.plot as xplt
from xarray import DataArray, Dataset
from xarray.namedarray.utils import module_available
from xarray.plot.dataarray_plot import _infer_interval_breaks
from xarray.plot.dataset_plot import _infer_meta_data
from xarray.plot.utils import (
from xarray.tests import (
@pytest.mark.slow
class TestImshow(Common2dMixin, PlotTestCase):
    plotfunc = staticmethod(xplt.imshow)

    @pytest.mark.xfail(reason='Failing inside matplotlib. Should probably be fixed upstream because other plot functions can handle it. Remove this test when it works, already in Common2dMixin')
    def test_dates_are_concise(self) -> None:
        import matplotlib.dates as mdates
        time = pd.date_range('2000-01-01', '2000-01-10')
        a = DataArray(np.random.randn(2, len(time)), [('xx', [1, 2]), ('t', time)])
        self.plotfunc(a, x='t')
        ax = plt.gca()
        assert isinstance(ax.xaxis.get_major_locator(), mdates.AutoDateLocator)
        assert isinstance(ax.xaxis.get_major_formatter(), mdates.ConciseDateFormatter)

    @pytest.mark.slow
    def test_imshow_called(self) -> None:
        assert not self.imshow_called(self.darray.plot.contourf)
        assert self.imshow_called(self.darray.plot.imshow)

    def test_xy_pixel_centered(self) -> None:
        self.darray.plot.imshow(yincrease=False)
        assert np.allclose([-0.5, 14.5], plt.gca().get_xlim())
        assert np.allclose([9.5, -0.5], plt.gca().get_ylim())

    def test_default_aspect_is_auto(self) -> None:
        self.darray.plot.imshow()
        assert 'auto' == plt.gca().get_aspect()

    @pytest.mark.slow
    def test_cannot_change_mpl_aspect(self) -> None:
        with pytest.raises(ValueError, match='not available in xarray'):
            self.darray.plot.imshow(aspect='equal')
        self.darray.plot.imshow(size=5, aspect=2)
        assert 'auto' == plt.gca().get_aspect()
        assert tuple(plt.gcf().get_size_inches()) == (10, 5)

    @pytest.mark.slow
    def test_primitive_artist_returned(self) -> None:
        artist = self.plotmethod()
        assert isinstance(artist, mpl.image.AxesImage)

    @pytest.mark.slow
    @requires_seaborn
    def test_seaborn_palette_needs_levels(self) -> None:
        with pytest.raises(ValueError):
            self.plotmethod(cmap='husl')

    def test_2d_coord_names(self) -> None:
        with pytest.raises(ValueError, match='requires 1D coordinates'):
            self.plotmethod(x='x2d', y='y2d')

    def test_plot_rgb_image(self) -> None:
        DataArray(easy_array((10, 15, 3), start=0), dims=['y', 'x', 'band']).plot.imshow()
        assert 0 == len(find_possible_colorbars())

    def test_plot_rgb_image_explicit(self) -> None:
        DataArray(easy_array((10, 15, 3), start=0), dims=['y', 'x', 'band']).plot.imshow(y='y', x='x', rgb='band')
        assert 0 == len(find_possible_colorbars())

    def test_plot_rgb_faceted(self) -> None:
        DataArray(easy_array((2, 2, 10, 15, 3), start=0), dims=['a', 'b', 'y', 'x', 'band']).plot.imshow(row='a', col='b')
        assert 0 == len(find_possible_colorbars())

    def test_plot_rgba_image_transposed(self) -> None:
        DataArray(easy_array((4, 10, 15), start=0), dims=['band', 'y', 'x']).plot.imshow()

    def test_warns_ambigious_dim(self) -> None:
        arr = DataArray(easy_array((3, 3, 3)), dims=['y', 'x', 'band'])
        with pytest.warns(UserWarning):
            arr.plot.imshow()
        arr.plot.imshow(rgb='band')
        arr.plot.imshow(x='x', y='y')

    def test_rgb_errors_too_many_dims(self) -> None:
        arr = DataArray(easy_array((3, 3, 3, 3)), dims=['y', 'x', 'z', 'band'])
        with pytest.raises(ValueError):
            arr.plot.imshow(rgb='band')

    def test_rgb_errors_bad_dim_sizes(self) -> None:
        arr = DataArray(easy_array((5, 5, 5)), dims=['y', 'x', 'band'])
        with pytest.raises(ValueError):
            arr.plot.imshow(rgb='band')

    @pytest.mark.parametrize(['vmin', 'vmax', 'robust'], [(-1, None, False), (None, 2, False), (-1, 1, False), (0, 0, False), (0, None, True), (None, -1, True)])
    def test_normalize_rgb_imshow(self, vmin: float | None, vmax: float | None, robust: bool) -> None:
        da = DataArray(easy_array((5, 5, 3), start=-0.6, stop=1.4))
        arr = da.plot.imshow(vmin=vmin, vmax=vmax, robust=robust).get_array()
        assert arr is not None
        assert 0 <= arr.min() <= arr.max() <= 1

    def test_normalize_rgb_one_arg_error(self) -> None:
        da = DataArray(easy_array((5, 5, 3), start=-0.6, stop=1.4))
        for vmin, vmax in ((None, -1), (2, None)):
            with pytest.raises(ValueError):
                da.plot.imshow(vmin=vmin, vmax=vmax)
        for vmin2, vmax2 in ((-1.2, -1), (2, 2.1)):
            da.plot.imshow(vmin=vmin2, vmax=vmax2)

    @pytest.mark.parametrize('dtype', [np.uint8, np.int8, np.int16])
    def test_imshow_rgb_values_in_valid_range(self, dtype) -> None:
        da = DataArray(np.arange(75, dtype=dtype).reshape((5, 5, 3)))
        _, ax = plt.subplots()
        out = da.plot.imshow(ax=ax).get_array()
        assert out is not None
        actual_dtype = out.dtype
        assert actual_dtype is not None
        assert actual_dtype == np.uint8
        assert (out[..., :3] == da.values).all()
        assert (out[..., -1] == 255).all()

    @pytest.mark.filterwarnings('ignore:Several dimensions of this array')
    def test_regression_rgb_imshow_dim_size_one(self) -> None:
        da = DataArray(easy_array((1, 3, 3), start=0.0, stop=1.0))
        da.plot.imshow()

    def test_origin_overrides_xyincrease(self) -> None:
        da = DataArray(easy_array((3, 2)), coords=[[-2, 0, 2], [-1, 1]])
        with figure_context():
            da.plot.imshow(origin='upper')
            assert plt.xlim()[0] < 0
            assert plt.ylim()[1] < 0
        with figure_context():
            da.plot.imshow(origin='lower')
            assert plt.xlim()[0] < 0
            assert plt.ylim()[0] < 0