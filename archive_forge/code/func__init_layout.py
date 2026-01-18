import param
from holoviews.plotting.util import attach_streams
from ...core import AdjointLayout, Empty, GridMatrix, GridSpace, HoloMap, NdLayout
from ...core.options import Store
from ...core.util import wrap_tuple
from ...element import Histogram
from ..plot import (
from .util import configure_matching_axes_from_dims, figure_grid
def _init_layout(self, layout):
    layout_count = 0
    collapsed_layout = layout.clone(shared_data=False, id=layout.id)
    frame_ranges = self.compute_ranges(layout, None, None)
    frame_ranges = dict([(key, self.compute_ranges(layout, key, frame_ranges)) for key in self.keys])
    layout_items = layout.grid_items()
    layout_dimensions = layout.kdims if isinstance(layout, NdLayout) else None
    layout_subplots, layouts, paths = ({}, {}, {})
    for r, c in self.coords:
        key, view = layout_items.get((c, r) if self.transpose else (r, c), (None, None))
        view = view if isinstance(view, AdjointLayout) else AdjointLayout([view])
        layouts[r, c] = view
        paths[r, c] = key
        layout_lens = {1: 'Single', 2: 'Dual', 3: 'Triple'}
        layout_type = layout_lens.get(len(view), 'Single')
        positions = AdjointLayoutPlot.layout_dict[layout_type]['positions']
        layout_key, _ = layout_items.get((r, c), (None, None))
        if isinstance(layout, NdLayout) and layout_key:
            layout_dimensions = dict(zip(layout_dimensions, layout_key))
        obj = layouts[r, c]
        empty = isinstance(obj.main, Empty)
        if empty:
            obj = AdjointLayout([])
        else:
            layout_count += 1
        subplot_data = self._create_subplots(obj, positions, layout_dimensions, frame_ranges, num=0 if empty else layout_count)
        subplots, adjoint_layout = subplot_data
        plotopts = self.lookup_options(view, 'plot').options
        layout_plot = AdjointLayoutPlot(adjoint_layout, layout_type, subplots, **plotopts)
        layout_subplots[r, c] = layout_plot
        if layout_key:
            collapsed_layout[layout_key] = adjoint_layout
    return (collapsed_layout, layout_subplots, paths)