import uuid
import warnings
from ast import literal_eval
from collections import Counter, defaultdict
from functools import partial
from itertools import groupby, product
import numpy as np
import param
from panel.config import config
from panel.io.document import unlocked
from panel.io.notebook import push
from panel.io.state import state
from pyviz_comms import JupyterComm
from ..core import traversal, util
from ..core.data import Dataset, disable_pipeline
from ..core.element import Element, Element3D
from ..core.layout import Empty, Layout, NdLayout
from ..core.options import Compositor, SkipRendering, Store, lookup_options
from ..core.overlay import CompositeOverlay, NdOverlay, Overlay
from ..core.spaces import DynamicMap, HoloMap
from ..core.util import isfinite, stream_parameters
from ..element import Graph, Table
from ..selection import NoOpSelectionDisplay
from ..streams import RangeX, RangeXY, RangeY, Stream
from ..util.transform import dim
from .util import (
def _create_subplot(self, key, obj, streams, ranges):
    registry = Store.registry[self.renderer.backend]
    ordering = util.layer_sort(self.hmap)
    overlay_type = 1 if self.hmap.type == Overlay else 2
    group_fn = lambda x: (x.type.__name__, x.last.group, x.last.label)
    opts = {'overlaid': overlay_type}
    if self.hmap.type == Overlay:
        style_key = (obj.type.__name__,) + key
        if self.overlay_dims:
            opts['overlay_dims'] = self.overlay_dims
    else:
        if not isinstance(key, tuple):
            key = (key,)
        style_key = group_fn(obj) + key
        opts['overlay_dims'] = dict(zip(self.hmap.last.kdims, key))
    if self.batched:
        vtype = type(obj.last.last)
        oidx = 0
    else:
        vtype = type(obj.last)
        if style_key not in ordering:
            ordering.append(style_key)
        oidx = ordering.index(style_key)
    plottype = registry.get(vtype, None)
    if plottype is None:
        self.param.warning('No plotting class for {} type and {} backend found. '.format(vtype.__name__, self.renderer.backend))
        return None
    length = self.style_grouping
    group_key = style_key[:length]
    zorder = self.zorder + oidx + self.zoffset
    cyclic_index = self.group_counter[group_key]
    self.cyclic_index_lookup[style_key] = cyclic_index
    self.group_counter[group_key] += 1
    group_length = self.map_lengths[group_key]
    if not isinstance(plottype, PlotSelector) and issubclass(plottype, GenericOverlayPlot):
        opts['group_counter'] = self.group_counter
        opts['show_legend'] = self.show_legend
        if not any((len(frame) for frame in obj)):
            self.param.warning('%s is empty and will be skipped during plotting' % obj.last)
            return None
    elif self.batched and 'batched' in plottype._plot_methods:
        param_vals = self.param.values()
        propagate = {opt: param_vals[opt] for opt in self._propagate_options if opt in param_vals}
        opts['batched'] = self.batched
        opts['overlaid'] = self.overlaid
        opts.update(propagate)
    if len(ordering) > self.legend_limit:
        opts['show_legend'] = False
    style = self.lookup_options(obj.last, 'style').max_cycles(group_length)
    passed_handles = {k: v for k, v in self.handles.items() if k in self._passed_handles}
    plotopts = dict(opts, cyclic_index=cyclic_index, invert_axes=self.invert_axes, dimensions=self.dimensions, keys=self.keys, layout_dimensions=self.layout_dimensions, ranges=ranges, show_title=self.show_title, style=style, uniform=self.uniform, fontsize=self.fontsize, streams=streams, renderer=self.renderer, adjoined=self.adjoined, stream_sources=self.stream_sources, projection=self.projection, fontscale=self.fontscale, zorder=zorder, root=self.root, **passed_handles)
    return plottype(obj, **plotopts)