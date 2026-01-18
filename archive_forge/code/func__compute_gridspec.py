from contextlib import contextmanager
from itertools import chain
import matplotlib as mpl
import numpy as np
import param
from matplotlib import (
from matplotlib.font_manager import font_scalings
from mpl_toolkits.mplot3d import Axes3D  # noqa (For 3D plots)
from ...core import (
from ...core.options import SkipRendering, Store
from ...core.util import int_to_alpha, int_to_roman, wrap_tuple_streams
from ..plot import (
from ..util import attach_streams, collate, displayable
from .util import compute_ratios, fix_aspect, get_old_rcparams
def _compute_gridspec(self, layout):
    """
        Computes the tallest and widest cell for each row and column
        by examining the Layouts in the GridSpace. The GridSpec is then
        instantiated and the LayoutPlots are configured with the
        appropriate embedded layout_types. The first element of the
        returned tuple is a dictionary of all the LayoutPlots indexed
        by row and column. The second dictionary in the tuple supplies
        the grid indices needed to instantiate the axes for each
        LayoutPlot.
        """
    layout_items = layout.grid_items()
    layout_dimensions = layout.kdims if isinstance(layout, NdLayout) else None
    layouts = {}
    col_widthratios, row_heightratios = ({}, {})
    for r, c in self.coords:
        _, view = layout_items.get((c, r) if self.transpose else (r, c), (None, None))
        if isinstance(view, NdLayout):
            raise SkipRendering('Cannot render NdLayout nested inside a Layout')
        layout_view = view if isinstance(view, AdjointLayout) else AdjointLayout([view])
        layouts[r, c] = layout_view
        layout_lens = {1: 'Single', 2: 'Dual', 3: 'Triple'}
        layout_type = layout_lens[len(layout_view)]
        main = layout_view.main
        main = main.last if isinstance(main, HoloMap) else main
        main_options = self.lookup_options(main, 'plot').options if main else {}
        if main and (not isinstance(main_options.get('aspect', 1), str)):
            main_aspect = np.nan if isinstance(main, Empty) else main_options.get('aspect', 1)
            main_aspect = self.aspect_weight * main_aspect + 1 - self.aspect_weight
        else:
            main_aspect = np.nan
        if layout_type in ['Dual', 'Triple']:
            el = layout_view.get('right', None)
            eltype = type(el)
            if el and eltype in MPLPlot.sideplots:
                plot_type = MPLPlot.sideplots[type(el)]
                ratio = 0.6 * (plot_type.subplot_size + plot_type.border_size)
                width_ratios = [4, 4 * ratio]
            else:
                width_ratios = [4, 1]
        else:
            width_ratios = [4]
        inv_aspect = 1.0 / main_aspect if main_aspect else np.nan
        if layout_type in ['Embedded Dual', 'Triple']:
            el = layout_view.get('top', None)
            eltype = type(el)
            if el and eltype in MPLPlot.sideplots:
                plot_type = MPLPlot.sideplots[type(el)]
                ratio = 0.6 * (plot_type.subplot_size + plot_type.border_size)
                height_ratios = [4 * ratio, 4]
            else:
                height_ratios = [1, 4]
        else:
            height_ratios = [4]
        if not isinstance(main_aspect, (str, type(None))):
            width_ratios = [wratio * main_aspect for wratio in width_ratios]
            height_ratios = [hratio * inv_aspect for hratio in height_ratios]
        layout_shape = (len(width_ratios), len(height_ratios))
        prev_heights = row_heightratios.get(r, (0, []))
        if layout_shape[1] > prev_heights[0]:
            row_heightratios[r] = [layout_shape[1], prev_heights[1]]
        row_heightratios[r][1].append(height_ratios)
        prev_widths = col_widthratios.get(c, (0, []))
        if layout_shape[0] > prev_widths[0]:
            col_widthratios[c] = (layout_shape[0], prev_widths[1])
        col_widthratios[c][1].append(width_ratios)
    col_splits = [v[0] for __, v in sorted(col_widthratios.items())]
    row_splits = [v[0] for ___, v in sorted(row_heightratios.items())]
    widths = np.array([r for col in col_widthratios.values() for ratios in col[1] for r in ratios]) / 4
    wr_unnormalized = compute_ratios(col_widthratios, False)
    hr_list = compute_ratios(row_heightratios)
    wr_list = compute_ratios(col_widthratios)
    cols, rows = (len(wr_list), len(hr_list))
    wr_list = [r if np.isfinite(r) else 1 for r in wr_list]
    hr_list = [r if np.isfinite(r) else 1 for r in hr_list]
    width = sum([r if np.isfinite(r) else 1 for r in wr_list])
    yscale = width / sum([1 / v * 4 if np.isfinite(v) else 4 for v in wr_unnormalized])
    if self.absolute_scaling:
        width = width * np.nanmax(widths)
    xinches, yinches = (None, None)
    if not isinstance(self.fig_inches, (tuple, list)):
        xinches = self.fig_inches * width
        yinches = xinches / yscale
    elif self.fig_inches[0] is None:
        xinches = self.fig_inches[1] * yscale
        yinches = self.fig_inches[1]
    elif self.fig_inches[1] is None:
        xinches = self.fig_inches[0]
        yinches = self.fig_inches[0] / yscale
    if xinches and yinches:
        self.handles['fig'].set_size_inches([xinches, yinches])
    self.gs = gridspec.GridSpec(rows, cols, width_ratios=wr_list, height_ratios=hr_list, wspace=self.hspace, hspace=self.vspace)
    self.handles['fig'].clf()
    gidx = 0
    layout_count = 0
    tight = self.tight
    collapsed_layout = layout.clone(shared_data=False, id=layout.id)
    frame_ranges = self.compute_ranges(layout, None, None)
    keys = self.keys[:1] if self.dynamic else self.keys
    frame_ranges = dict([(key, self.compute_ranges(layout, key, frame_ranges)) for key in keys])
    layout_subplots, layout_axes = ({}, {})
    for r, c in self.coords:
        wsplits = col_splits[c]
        hsplits = row_splits[r]
        if (wsplits, hsplits) == (1, 1):
            layout_type = 'Single'
        elif (wsplits, hsplits) == (2, 1):
            layout_type = 'Dual'
        elif (wsplits, hsplits) == (1, 2):
            layout_type = 'Embedded Dual'
        elif (wsplits, hsplits) == (2, 2):
            layout_type = 'Triple'
        view = layouts[r, c]
        positions = AdjointLayoutPlot.layout_dict[layout_type]
        _, _, projs = self._create_subplots(layouts[r, c], positions, None, frame_ranges, create=False)
        gidx, gsinds = self.grid_situate(gidx, layout_type, cols)
        layout_key, _ = layout_items.get((r, c), (None, None))
        if isinstance(layout, NdLayout) and layout_key:
            layout_dimensions = dict(zip(layout_dimensions, layout_key))
        obj = layouts[r, c]
        empty = isinstance(obj.main, Empty)
        if view.main is None:
            continue
        elif empty:
            obj = AdjointLayout([])
        elif not view.traverse(lambda x: x, [Element]):
            self.param.warning(f'{obj.main} is empty, skipping subplot.')
            continue
        elif self.transpose:
            layout_count = c * self.rows + (r + 1)
        else:
            layout_count += 1
        subaxes = [plt.subplot(self.gs[ind], projection=proj) for ind, proj in zip(gsinds, projs)]
        subplot_data = self._create_subplots(obj, positions, layout_dimensions, frame_ranges, dict(zip(positions, subaxes)), num=0 if empty else layout_count)
        subplots, adjoint_layout, _ = subplot_data
        layout_axes[r, c] = subaxes
        plotopts = self.lookup_options(view, 'plot').options
        layout_plot = AdjointLayoutPlot(adjoint_layout, layout_type, subaxes, subplots, fig=self.handles['fig'], **plotopts)
        layout_subplots[r, c] = layout_plot
        tight = not any((type(p) is GridPlot for p in layout_plot.subplots.values())) and tight
        if layout_key:
            collapsed_layout[layout_key] = adjoint_layout
    if tight:
        if isinstance(self.tight_padding, (tuple, list)):
            wpad, hpad = self.tight_padding
            padding = dict(w_pad=wpad, h_pad=hpad)
        else:
            padding = dict(w_pad=self.tight_padding, h_pad=self.tight_padding)
        self.gs.tight_layout(self.handles['fig'], rect=self.fig_bounds, **padding)
    return (layout_subplots, layout_axes, collapsed_layout)