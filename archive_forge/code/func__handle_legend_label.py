from __future__ import annotations
import logging # isort:skip
import numpy as np
from ..core.properties import field, value
from ..models import Legend, LegendItem
from ..util.strings import nice_join
def _handle_legend_label(label, legend, glyph_renderer):
    if not isinstance(label, str):
        raise ValueError('legend_label value must be a string')
    label = value(label)
    item = _find_legend_item(label, legend)
    if item:
        item.renderers.append(glyph_renderer)
    else:
        new_item = LegendItem(label=label, renderers=[glyph_renderer])
        legend.items.append(new_item)