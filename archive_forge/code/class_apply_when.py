import warnings
import numpy as np
import param
from packaging.version import Version
from param import _is_number
from ..core import (
from ..core.data import ArrayInterface, DictInterface, PandasInterface, default_datatype
from ..core.data.util import dask_array_module
from ..core.util import (
from ..element.chart import Histogram, Scatter
from ..element.path import Contours, Polygons
from ..element.raster import RGB, Image
from ..element.util import categorical_aggregate2d  # noqa (API import)
from ..streams import RangeXY
from ..util.locator import MaxNLocator
class apply_when(param.ParameterizedFunction):
    """
    Applies a selection depending on the current zoom range. If the
    supplied predicate function returns a True it will apply the
    operation otherwise it will return the raw element after the
    selection. For example the following will apply datashading if
    the number of points in the current viewport exceed 1000 otherwise
    just returning the selected points element:

       apply_when(points, operation=datashade, predicate=lambda x: x > 1000)
    """
    operation = param.Callable(default=lambda x: x)
    predicate = param.Callable(default=None)

    def _apply(self, element, x_range, y_range, invert=False):
        selected = element
        if x_range is not None and y_range is not None:
            selected = element[x_range, y_range]
        condition = self.predicate(selected)
        if not invert and condition or (invert and (not condition)):
            return selected
        elif selected.interface.gridded:
            return selected.clone([])
        else:
            return selected.iloc[:0]

    def __call__(self, obj, **params):
        if 'streams' in params:
            streams = params.pop('streams')
        else:
            streams = [RangeXY()]
        self.param.update(**params)
        if not self.predicate:
            raise ValueError('Must provide a predicate function to determine when to apply the operation and when to return the selected data.')
        applied = self.operation(obj.apply(self._apply, streams=streams))
        raw = obj.apply(self._apply, streams=streams, invert=True)
        return applied * raw