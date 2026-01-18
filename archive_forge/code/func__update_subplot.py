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
def _update_subplot(self, subplot, spec):
    """
        Updates existing subplots when the subplot has been assigned
        to plot an element that is not an exact match to the object
        it was initially assigned.
        """
    if spec in self.cyclic_index_lookup:
        cyclic_index = self.cyclic_index_lookup[spec]
    else:
        group_key = spec[:self.style_grouping]
        self.group_counter[group_key] += 1
        cyclic_index = self.group_counter[group_key]
        self.cyclic_index_lookup[spec] = cyclic_index
    subplot.cyclic_index = cyclic_index
    if subplot.overlay_dims:
        odim_key = util.wrap_tuple(spec[-1])
        new_dims = zip(subplot.overlay_dims, odim_key)
        subplot.overlay_dims = dict(new_dims)