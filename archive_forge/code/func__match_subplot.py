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
def _match_subplot(self, key, subplot, items, element):
    found = False
    temp_items = list(items)
    while not found:
        idx, spec, exact = dynamic_update(self, subplot, key, element, temp_items)
        if idx is not None:
            if not exact:
                exact_matches = [dynamic_update(self, subplot, k, element, temp_items) for k in self.subplots]
                exact_matches = [m for m in exact_matches if m[-1]]
                if exact_matches:
                    idx = exact_matches[0][0]
                    _, el = temp_items.pop(idx)
                    continue
        found = True
    if idx is not None:
        idx = items.index(temp_items.pop(idx))
    return (idx, spec, exact)