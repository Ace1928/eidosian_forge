import bisect
import re
import traceback
import warnings
from collections import defaultdict, namedtuple
import numpy as np
import param
from packaging.version import Version
from ..core import (
from ..core.ndmapping import item_check
from ..core.operation import Operation
from ..core.options import CallbackError, Cycle
from ..core.spaces import get_nested_streams
from ..core.util import (
from ..element import Points
from ..streams import LinkedStream, Params
from ..util.transform import dim
class categorical_legend(Operation):
    """
    Generates a Points element which contains information for generating
    a legend by inspecting the pipeline of a datashaded RGB element.
    """
    backend = param.String()

    def _process(self, element, key=None):
        import datashader as ds
        from ..operation.datashader import datashade, rasterize, shade
        rasterize_op = element.pipeline.find(rasterize, skip_nonlinked=False)
        if isinstance(rasterize_op, datashade):
            shade_op = rasterize_op
        else:
            shade_op = element.pipeline.find(shade, skip_nonlinked=False)
        if None in (shade_op, rasterize_op):
            return None
        hvds = element.dataset
        input_el = element.pipeline.operations[0](hvds)
        agg = rasterize_op._get_aggregator(input_el, rasterize_op.aggregator)
        if not isinstance(agg, (ds.count_cat, ds.by)):
            return
        column = agg.column
        if hasattr(hvds.data, 'dtypes'):
            try:
                cats = list(hvds.data.dtypes[column].categories)
            except TypeError:
                cats = list(hvds.data.dtypes[column].categories.to_pandas())
            if cats == ['__UNKNOWN_CATEGORIES__']:
                cats = list(hvds.data[column].cat.as_known().categories)
        else:
            cats = list(hvds.dimension_values(column, expanded=False))
        colors = shade_op.color_key or ds.colors.Sets1to3
        color_data = [(0, 0, cat) for cat in cats]
        if isinstance(colors, list):
            cat_colors = {cat: colors[i] for i, cat in enumerate(cats)}
        else:
            cat_colors = {cat: colors[cat] for cat in cats}
        cmap = {}
        for cat, color in cat_colors.items():
            if isinstance(color, tuple):
                color = rgb2hex([v / 256 for v in color[:3]])
            cmap[cat] = color
        return Points(color_data, vdims=['category']).opts(cmap=cmap, color='category', show_legend=True, backend=self.p.backend, visible=False)