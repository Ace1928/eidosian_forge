import warnings
from collections.abc import Callable, Iterable
from functools import partial
import dask.dataframe as dd
import datashader as ds
import datashader.reductions as rd
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd
import param
import xarray as xr
from datashader.colors import color_lookup
from packaging.version import Version
from param.parameterized import bothmethod
from ..core import (
from ..core.data import (
from ..core.util import (
from ..element import (
from ..element.util import connect_tri_edges_pd
from ..streams import PointerXY
from .resample import LinkableOperation, ResampleOperation2D
class inspect_mask(Operation):
    """
    Operation used to display the inspection mask, for use with other
    inspection operations. Can be used directly but is more commonly
    constructed using the mask property of the corresponding inspector
    operation.
    """
    pixels = param.Integer(default=3, doc='\n       Size of the mask that should match the pixels parameter used in\n       the associated inspection operation.')
    streams = param.ClassSelector(default=[PointerXY], class_=(dict, list))
    x = param.Number(default=0)
    y = param.Number(default=0)

    @classmethod
    def _distance_args(cls, element, x_range, y_range, pixels):
        ycount, xcount = element.interface.shape(element, gridded=True)
        x_delta = abs(x_range[1] - x_range[0]) / xcount
        y_delta = abs(y_range[1] - y_range[0]) / ycount
        return (x_delta * pixels, y_delta * pixels)

    def _process(self, raster, key=None):
        if isinstance(raster, RGB):
            raster = raster[..., raster.vdims[-1]]
        x_range, y_range = (raster.range(0), raster.range(1))
        xdelta, ydelta = self._distance_args(raster, x_range, y_range, self.p.pixels)
        x, y = (self.p.x, self.p.y)
        return self._indicator(raster.kdims, x, y, xdelta, ydelta)

    def _indicator(self, kdims, x, y, xdelta, ydelta):
        rect = np.array([(x - xdelta / 2, y - ydelta / 2), (x + xdelta / 2, y - ydelta / 2), (x + xdelta / 2, y + ydelta / 2), (x - xdelta / 2, y + ydelta / 2)])
        data = {(str(kdims[0]), str(kdims[1])): rect}
        return Polygons(data, kdims=kdims)