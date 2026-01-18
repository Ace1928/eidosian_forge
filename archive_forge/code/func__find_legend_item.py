from __future__ import annotations
import logging # isort:skip
import numpy as np
from ..core.properties import field, value
from ..models import Legend, LegendItem
from ..util.strings import nice_join
def _find_legend_item(label, legend):
    for item in legend.items:
        if item.label == label:
            return item
    return None