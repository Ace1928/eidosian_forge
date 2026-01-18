from __future__ import annotations
import logging # isort:skip
import numpy as np
from ..core.properties import field, value
from ..models import Legend, LegendItem
from ..util.strings import nice_join
def _get_or_create_legend(plot):
    panels = plot.above + plot.below + plot.left + plot.right + plot.center
    legends = [obj for obj in panels if isinstance(obj, Legend)]
    if not legends:
        legend = Legend()
        plot.add_layout(legend)
        return legend
    if len(legends) == 1:
        return legends[0]
    raise RuntimeError('Plot %s configured with more than one legend renderer, cannot use legend_* convenience arguments' % plot)