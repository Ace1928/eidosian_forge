import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
@pytest.mark.usefixtures('load_xarray_accessor')
class TestXarrayTitle:

    def test_dataarray_2d_with_title(self, da, backend):
        da_sel = da.sel(time=0, band=0)
        plot = da_sel.hvplot()
        opts = Store.lookup_options(backend, plot, 'plot')
        assert opts.kwargs['title'] == 'time = 0, band = 0'

    def test_dataarray_1d_with_title(self, da, backend):
        da_sel = da.sel(time=0, band=0, x=0)
        plot = da_sel.hvplot()
        opts = Store.lookup_options(backend, plot, 'plot')
        assert opts.kwargs['title'] == 'time = 0, x = 0, band = 0'

    def test_dataarray_1d_and_by_with_title(self, da, backend):
        da_sel = da.sel(time=0, band=0, x=[0, 1])
        plot = da_sel.hvplot(by='x')
        opts = Store.lookup_options(backend, plot, 'plot')
        assert opts.kwargs['title'] == 'time = 0, band = 0'

    def test_override_title(self, da, backend):
        da_sel = da.sel(time=0, band=0)
        plot = da_sel.hvplot(title='title')
        opts = Store.lookup_options(backend, plot, 'plot')
        assert opts.kwargs['title'] == 'title'

    def test_dataarray_4d_line_no_title(self, da, backend):
        plot = da.hvplot.line(dynamic=False)
        opts = Store.lookup_options(backend, plot.last, 'plot')
        assert 'title' not in opts.kwargs

    def test_dataarray_3d_histogram_with_title(self, da, backend):
        da_sel = da.sel(time=0)
        plot = da_sel.hvplot()
        opts = Store.lookup_options(backend, plot, 'plot')
        assert opts.kwargs['title'] == 'time = 0'

    def test_dataset_empty_raises(self, ds1, backend):
        with pytest.raises(ValueError, match='empty xarray.Dataset'):
            ds1.drop_vars('foo').hvplot()

    def test_dataset_one_var_behaves_like_dataarray(self, ds1, backend):
        ds_sel = ds1.sel(time=0, band=0)
        plot = ds_sel.hvplot()
        opts = Store.lookup_options(backend, plot, 'plot')
        assert opts.kwargs['title'] == 'time = 0, band = 0'

    def test_dataset_scatter_with_title(self, ds2, backend):
        ds_sel = ds2.sel(time=0, band=0, x=0, y=0)
        plot = ds_sel.hvplot.scatter(x='foo', y='bar')
        opts = Store.lookup_options(backend, plot, 'plot')
        assert opts.kwargs['title'] == 'time = 0, y = 0, x = 0, band = 0'