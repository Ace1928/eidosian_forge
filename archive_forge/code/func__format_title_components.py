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
def _format_title_components(self, key, dimensions=True, separator='\n'):
    dim_title = self._frame_title(key, 3, separator) if dimensions else ''
    layout = self.layout
    type_name = type(self.layout).__name__
    group = util.bytes_to_unicode(layout.group if layout.group != type_name else '')
    label = util.bytes_to_unicode(layout.label)
    return (label, group, type_name, dim_title)