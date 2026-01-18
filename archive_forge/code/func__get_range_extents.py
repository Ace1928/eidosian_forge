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
def _get_range_extents(self, element, ranges, range_type, xdim, ydim, zdim):
    dims = element.dimensions()
    ndims = len(dims)
    xdim = xdim or (dims[0] if ndims else None)
    ydim = ydim or (dims[1] if ndims > 1 else None)
    if isinstance(self.projection, str) and self.projection == '3d':
        zdim = zdim or (dims[2] if ndims > 2 else None)
    else:
        zdim = None
    (x0, x1), xsrange, xhrange = get_range(element, ranges, xdim)
    (y0, y1), ysrange, yhrange = get_range(element, ranges, ydim)
    (z0, z1), zsrange, zhrange = get_range(element, ranges, zdim)
    trigger = False
    if not self.overlaid and (not self.batched):
        xspan, yspan, zspan = (v / 2.0 for v in get_axis_padding(self.default_span))
        mx0, mx1 = get_minimum_span(x0, x1, xspan)
        if x0 != mx0 or x1 != mx1:
            x0, x1 = (mx0, mx1)
            trigger = True
        my0, my1 = get_minimum_span(y0, y1, yspan)
        if y0 != my0 or y1 != my1:
            y0, y1 = (my0, my1)
            trigger = True
        mz0, mz1 = get_minimum_span(z0, z1, zspan)
    xpad, ypad, zpad = self.get_padding(element, (x0, y0, z0, x1, y1, z1))
    if range_type == 'soft':
        x0, x1 = xsrange
    elif range_type == 'hard':
        x0, x1 = xhrange
    elif xdim == 'categorical':
        x0, x1 = ('', '')
    elif range_type == 'combined':
        x0, x1 = util.dimension_range(x0, x1, xhrange, xsrange, xpad, self.logx)
    if range_type == 'soft':
        y0, y1 = ysrange
    elif range_type == 'hard':
        y0, y1 = yhrange
    elif range_type == 'combined':
        y0, y1 = util.dimension_range(y0, y1, yhrange, ysrange, ypad, self.logy)
    elif ydim == 'categorical':
        y0, y1 = ('', '')
    elif ydim is None:
        y0, y1 = (np.nan, np.nan)
    if isinstance(self.projection, str) and self.projection == '3d':
        if range_type == 'soft':
            z0, z1 = zsrange
        elif range_type == 'data':
            z0, z1 = zhrange
        elif range_type == 'combined':
            z0, z1 = util.dimension_range(z0, z1, zhrange, zsrange, zpad, self.logz)
        elif zdim == 'categorical':
            z0, z1 = ('', '')
        elif zdim is None:
            z0, z1 = (np.nan, np.nan)
        return (x0, y0, z0, x1, y1, z1)
    if not self.drawn:
        for stream in getattr(self, 'source_streams', []):
            if isinstance(stream, (RangeX, RangeY, RangeXY)) and trigger and (stream not in self._trigger):
                self._trigger.append(stream)
    return (x0, y0, x1, y1)