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
def _create_dynamic_subplots(self, key, items, ranges, **init_kwargs):
    """
        Handles the creation of new subplots when a DynamicMap returns
        a changing set of elements in an Overlay.
        """
    length = self.style_grouping
    group_fn = lambda x: (x.type.__name__, x.last.group, x.last.label)
    for k, obj in items:
        vmap = self.hmap.clone([(key, obj)])
        self.map_lengths[group_fn(vmap)[:length]] += 1
        subplot = self._create_subplot(k, vmap, [], ranges)
        if subplot is None:
            continue
        subplot.document = self.document
        if self.comm:
            subplot.comm = self.comm
        self.subplots[k] = subplot
        subplot.initialize_plot(ranges, **init_kwargs)
        subplot.update_frame(key, ranges, element=obj)
        self.dynamic_subplots.append(subplot)