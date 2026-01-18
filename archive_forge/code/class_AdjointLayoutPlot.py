import param
from holoviews.plotting.util import attach_streams
from ...core import AdjointLayout, Empty, GridMatrix, GridSpace, HoloMap, NdLayout
from ...core.options import Store
from ...core.util import wrap_tuple
from ...element import Histogram
from ..plot import (
from .util import configure_matching_axes_from_dims, figure_grid
class AdjointLayoutPlot(PlotlyPlot, GenericAdjointLayoutPlot):
    registry = {}

    def __init__(self, layout, layout_type, subplots, **params):
        self.layout = layout
        self.layout_type = layout_type
        self.view_positions = self.layout_dict[self.layout_type]['positions']
        super().__init__(subplots=subplots, **params)

    def initialize_plot(self, ranges=None, is_geo=False):
        """
        Plot all the views contained in the AdjointLayout Object using axes
        appropriate to the layout configuration. All the axes are
        supplied by LayoutPlot - the purpose of the call is to
        invoke subplots with correct options and styles and hide any
        empty axes as necessary.
        """
        return self.generate_plot(self.keys[-1], ranges, is_geo=is_geo)

    def generate_plot(self, key, ranges=None, is_geo=False):
        adjoined_plots = []
        for pos in ['main', 'right', 'top']:
            subplot = self.subplots.get(pos, None)
            if subplot:
                adjoined_plots.append(subplot.generate_plot(key, ranges=ranges, is_geo=is_geo))
        if not adjoined_plots:
            adjoined_plots = [None]
        return adjoined_plots