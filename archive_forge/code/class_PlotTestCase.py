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
@requires_matplotlib
class PlotTestCase:

    @pytest.fixture(autouse=True)
    def setup(self):
        yield
        plt.close('all')

    def pass_in_axis(self, plotmethod, subplot_kw=None):
        fig, axs = plt.subplots(ncols=2, subplot_kw=subplot_kw)
        plotmethod(ax=axs[0])
        assert axs[0].has_data()

    @pytest.mark.slow
    def imshow_called(self, plotmethod):
        plotmethod()
        images = plt.gca().findobj(mpl.image.AxesImage)
        return len(images) > 0

    def contourf_called(self, plotmethod):
        plotmethod()

        def matchfunc(x):
            return isinstance(x, (mpl.collections.PathCollection, mpl.contour.QuadContourSet))
        paths = plt.gca().findobj(matchfunc)
        return len(paths) > 0