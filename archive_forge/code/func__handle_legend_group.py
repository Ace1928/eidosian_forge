from __future__ import annotations
import logging # isort:skip
import numpy as np
from ..core.properties import field, value
from ..models import Legend, LegendItem
from ..util.strings import nice_join
def _handle_legend_group(label, legend, glyph_renderer):
    if not isinstance(label, str):
        raise ValueError('legend_group value must be a string')
    source = glyph_renderer.data_source
    if source is None:
        raise ValueError("Cannot use 'legend_group' on a glyph without a data source already configured")
    if not (hasattr(source, 'column_names') and label in source.column_names):
        raise ValueError('Column to be grouped does not exist in glyph data source')
    column = source.data[label]
    vals, inds = np.unique(column, return_index=1)
    for val, ind in zip(vals, inds):
        label = value(str(val))
        new_item = LegendItem(label=label, renderers=[glyph_renderer], index=ind)
        legend.items.append(new_item)