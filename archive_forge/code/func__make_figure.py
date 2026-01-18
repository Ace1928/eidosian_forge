from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from .._utils import add_margins, cross_join, join_keys, match, ninteraction
from ..exceptions import PlotnineError
from .facet import (
from .strips import Strips, strip
def _make_figure(self):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    layout = self.layout
    space = self.space
    ratios = {}
    if isinstance(space, str):
        if space in {'free', 'free_x'}:
            pidx: list[int] = layout.layout.sort_values('COL').drop_duplicates('COL').index.tolist()
            panel_views = [layout.panel_params[i] for i in pidx]
            ratios['width_ratios'] = [np.ptp(pv.x.range) for pv in panel_views]
        if space in {'free', 'free_y'}:
            pidx = layout.layout.sort_values('ROW').drop_duplicates('ROW').index.tolist()
            panel_views = [layout.panel_params[i] for i in pidx]
            ratios['height_ratios'] = [np.ptp(pv.y.range) for pv in panel_views]
    if isinstance(self.space, dict):
        if len(self.space['x']) != self.ncol:
            raise ValueError('The number of x-ratios for the facet space sizes should match the number of columns.')
        if len(self.space['y']) != self.nrow:
            raise ValueError('The number of y-ratios for the facet space sizes should match the number of rows.')
        ratios['width_ratios'] = self.space.get('x')
        ratios['height_ratios'] = self.space.get('y')
    return (plt.figure(), GridSpec(self.nrow, self.ncol, **ratios))