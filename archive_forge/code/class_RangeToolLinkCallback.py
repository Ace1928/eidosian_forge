import numpy as np
from bokeh.models import CustomJS, Toolbar
from bokeh.models.tools import RangeTool
from ...core.util import isscalar
from ..links import (
from ..plot import GenericElementPlot, GenericOverlayPlot
class RangeToolLinkCallback(LinkCallback):
    """
    Attaches a RangeTool to the source plot and links it to the
    specified axes on the target plot
    """

    def __init__(self, root_model, link, source_plot, target_plot):
        toolbars = list(root_model.select({'type': Toolbar}))
        axes = {}
        for axis in ('x', 'y'):
            if axis not in link.axes:
                continue
            axes[f'{axis}_range'] = target_plot.handles[f'{axis}_range']
            bounds = getattr(link, f'bounds{axis}', None)
            if bounds is None:
                continue
            start, end = bounds
            if start is not None:
                axes[f'{axis}_range'].start = start
                axes[f'{axis}_range'].reset_start = start
            if end is not None:
                axes[f'{axis}_range'].end = end
                axes[f'{axis}_range'].reset_end = end
        tool = RangeTool(**axes)
        source_plot.state.add_tools(tool)
        if toolbars:
            toolbars[0].tools.append(tool)