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
class inspect(Operation):
    """
    Generalized inspect operation that detects the appropriate indicator
    type.
    """
    pixels = param.Integer(default=3, doc="\n       Number of pixels in data space around the cursor point to search\n       for hits in. The hit within this box mask that is closest to the\n       cursor's position is displayed.")
    null_value = param.Number(default=0, doc='\n       Value of raster which indicates no hits. For instance zero for\n       count aggregator (default) and commonly NaN for other (float)\n       aggregators. For RGBA images, the alpha channel is used which means\n       zero alpha acts as the null value.')
    value_bounds = param.NumericTuple(default=None, length=2, allow_None=True, doc='\n       If not None, a numeric bounds for the pixel under the cursor in\n       order for hits to be computed. Useful for count aggregators where\n       a value of (1,1000) would make sure no more than a thousand\n       samples will be searched.')
    hits = param.DataFrame(default=pd.DataFrame(), allow_None=True)
    max_indicators = param.Integer(default=1, doc='\n       Maximum number of indicator elements to display within the mask\n       of size pixels. Points are prioritized by distance from the\n       cursor point. This means that the default value of one shows the\n       single closest sample to the cursor. Note that this limit is not\n       applies to the hits parameter.')
    transform = param.Callable(default=identity, doc='\n      Function that transforms the hits dataframe before it is passed to\n      the Points element. Can be used to customize the value dimensions\n      e.g. to implement custom hover behavior.')
    streams = param.ClassSelector(default=dict(x=PointerXY.param.x, y=PointerXY.param.y), class_=(dict, list))
    x = param.Number(default=0, doc='x-position to inspect.')
    y = param.Number(default=0, doc='y-position to inspect.')
    _dispatch = {}

    @property
    def mask(self):
        return inspect_mask.instance(pixels=self.p.pixels)

    def _update_hits(self, event):
        self.hits = event.obj.hits

    @bothmethod
    def instance(self_or_cls, **params):
        inst = super().instance(**params)
        inst._op = None
        return inst

    def _process(self, raster, key=None):
        input_type = self._get_input_type(raster.pipeline.operations)
        inspect_operation = self._dispatch[input_type]
        if self._op is None:
            self._op = inspect_operation.instance()
            self._op.param.watch(self._update_hits, 'hits')
        elif not isinstance(self._op, inspect_operation):
            raise ValueError('Cannot reuse inspect instance on different datashader input type.')
        self._op.p = self.p
        return self._op._process(raster)

    def _get_input_type(self, operations):
        for op in operations:
            output_type = getattr(op, 'output_type', None)
            if output_type is not None:
                if output_type in [el[0] for el in rasterize._transforms]:
                    if issubclass(output_type, (Image, RGB)):
                        continue
                    return output_type
        raise RuntimeError('Could not establish input element type for datashader pipeline in the inspect operation.')