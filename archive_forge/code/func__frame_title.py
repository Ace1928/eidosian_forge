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
def _frame_title(self, key, group_size=2, separator='\n'):
    """
        Returns the formatted dimension group strings
        for a particular frame.
        """
    if self.layout_dimensions is not None:
        dimensions, key = zip(*self.layout_dimensions.items())
    elif not self.dynamic and (not self.uniform or len(self) == 1) or self.subplot:
        return ''
    else:
        key = key if isinstance(key, tuple) else (key,)
        dimensions = self.dimensions
    dimension_labels = [dim.pprint_value_string(k) for dim, k in zip(dimensions, key)]
    groups = [', '.join(dimension_labels[i * group_size:(i + 1) * group_size]) for i in range(len(dimension_labels))]
    return util.bytes_to_unicode(separator.join((g for g in groups if g)))