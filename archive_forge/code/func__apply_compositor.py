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
def _apply_compositor(self, holomap, ranges=None, keys=None, dimensions=None):
    """
        Given a HoloMap compute the appropriate (mapwise or framewise)
        ranges in order to apply the Compositor collapse operations in
        display mode (data collapse should already have happened).
        """
    defaultdim = holomap.ndims == 1 and holomap.kdims[0].name != 'Frame'
    if keys and ranges and dimensions and (not defaultdim):
        dim_inds = [dimensions.index(d) for d in holomap.kdims]
        sliced_keys = [tuple((k[i] for i in dim_inds)) for k in keys]
        frame_ranges = dict([(slckey, self.compute_ranges(holomap, key, ranges[key])) for key, slckey in zip(keys, sliced_keys) if slckey in holomap.data.keys()])
    else:
        mapwise_ranges = self.compute_ranges(holomap, None, None)
        frame_ranges = dict([(key, self.compute_ranges(holomap, key, mapwise_ranges)) for key in holomap.data.keys()])
    ranges = frame_ranges.values()
    with disable_pipeline():
        collapsed = Compositor.collapse(holomap, (ranges, frame_ranges.keys()), mode='display')
    return collapsed