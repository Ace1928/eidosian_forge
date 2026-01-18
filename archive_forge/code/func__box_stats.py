from collections import defaultdict
from functools import partial
import numpy as np
import param
from bokeh.models import Circle, FactorRange, HBar, VBar
from ...core import NdOverlay
from ...core.dimension import Dimension, Dimensioned
from ...core.ndmapping import sorted_context
from ...core.util import (
from ...operation.stats import univariate_kde
from ...util.transform import dim
from ..mixins import MultiDistributionMixin
from .chart import AreaPlot
from .element import ColorbarPlot, CompositeElementPlot, LegendPlot
from .path import PolygonPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import base_properties, fill_properties, line_properties
from .util import decode_bytes
def _box_stats(self, vals):
    is_finite = isfinite
    is_dask = is_dask_array(vals)
    is_cupy = is_cupy_array(vals)
    if is_cupy:
        import cupy
        percentile = cupy.percentile
        is_finite = cupy.isfinite
    elif is_dask:
        import dask.array as da
        percentile = da.percentile
    else:
        percentile = np.percentile
    vals = vals[is_finite(vals)]
    if is_dask or len(vals):
        q1, q2, q3 = (percentile(vals, q=q) for q in range(25, 100, 25))
        iqr = q3 - q1
        upper = max(vals[vals <= q3 + 1.5 * iqr].max(), q3)
        lower = min(vals[vals >= q1 - 1.5 * iqr].min(), q1)
    else:
        q1, q2, q3 = (0, 0, 0)
        upper, lower = (0, 0)
    outliers = vals[(vals > upper) | (vals < lower)]
    if is_cupy:
        return (q1.item(), q2.item(), q3.item(), upper.item(), lower.item(), cupy.asnumpy(outliers))
    elif is_dask:
        return da.compute(q1, q2, q3, upper, lower, outliers)
    else:
        return (q1, q2, q3, upper, lower, outliers)