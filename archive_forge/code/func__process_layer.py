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
def _process_layer(self, element, key=None):
    INTERPOLATE_FUNCS = {'steps-pre': self.pts_to_prestep, 'steps-mid': self.pts_to_midstep, 'steps-post': self.pts_to_poststep}
    if self.p.interpolation not in INTERPOLATE_FUNCS:
        return element
    x = element.dimension_values(0)
    is_datetime = isdatetime(x)
    if is_datetime:
        dt_type = 'datetime64[ns]'
        x = x.astype(dt_type)
    dvals = tuple((element.dimension_values(d) for d in element.dimensions()[1:]))
    xs, dvals = INTERPOLATE_FUNCS[self.p.interpolation](x, dvals)
    if is_datetime:
        xs = xs.astype(dt_type)
    return element.clone((xs,) + dvals)