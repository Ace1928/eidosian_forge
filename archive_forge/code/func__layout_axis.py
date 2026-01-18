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
def _layout_axis(self, layout, axis):
    fig = self.handles['fig']
    axkwargs = {'gid': str(self.position)} if axis else {}
    layout_axis = fig.add_subplot(1, 1, 1, **axkwargs)
    if axis:
        axis.set_visible(False)
        layout_axis.set_position(self.position)
    layout_axis.patch.set_visible(False)
    for ax, ax_obj in zip(['x', 'y'], [layout_axis.xaxis, layout_axis.yaxis]):
        tick_fontsize = self._fontsize(f'{ax}ticks', 'labelsize', common=False)
        if tick_fontsize:
            ax_obj.set_tick_params(**tick_fontsize)
    layout_axis.set_xlabel(layout.kdims[0].pprint_label, **self._fontsize('xlabel'))
    if layout.ndims == 2:
        layout_axis.set_ylabel(layout.kdims[1].pprint_label, **self._fontsize('ylabel'))
    dims = layout.kdims
    keys = layout.keys()
    if layout.ndims == 1:
        dim1_keys = keys
        dim2_keys = [0]
        layout_axis.get_yaxis().set_visible(False)
    else:
        dim1_keys, dim2_keys = zip(*keys)
        layout_axis.set_ylabel(dims[1].pprint_label)
        layout_axis.set_aspect(float(self.rows) / self.cols)
    plot_width = (1.0 - self.padding) / self.cols
    border_width = self.padding / (self.cols - 1) if self.cols > 1 else 0
    xticks = [plot_width / 2 + r * (plot_width + border_width) for r in range(self.cols)]
    plot_height = (1.0 - self.padding) / self.rows
    border_height = self.padding / (self.rows - 1) if layout.ndims > 1 else 0
    yticks = [plot_height / 2 + r * (plot_height + border_height) for r in range(self.rows)]
    layout_axis.set_xticks(xticks)
    layout_axis.set_xticklabels([dims[0].pprint_value(l) for l in sorted(set(dim1_keys))])
    for tick in layout_axis.get_xticklabels():
        tick.set_rotation(self.xrotation)
    ydim = dims[1] if layout.ndims > 1 else None
    layout_axis.set_yticks(yticks)
    layout_axis.set_yticklabels([ydim.pprint_value(l) if ydim else '' for l in sorted(set(dim2_keys))])
    for tick in layout_axis.get_yticklabels():
        tick.set_rotation(self.yrotation)
    if not self.show_frame:
        layout_axis.spines['right' if self.yaxis == 'left' else 'left'].set_visible(False)
        layout_axis.spines['bottom' if self.xaxis == 'top' else 'top'].set_visible(False)
    axis = layout_axis
    if self.xaxis is not None:
        axis.xaxis.set_ticks_position(self.xaxis)
        axis.xaxis.set_label_position(self.xaxis)
    else:
        axis.xaxis.set_visible(False)
    if self.yaxis is not None:
        axis.yaxis.set_ticks_position(self.yaxis)
        axis.yaxis.set_label_position(self.yaxis)
    else:
        axis.yaxis.set_visible(False)
    for pos in ['left', 'right', 'top', 'bottom']:
        axis.spines[pos].set_visible(False)
    return layout_axis