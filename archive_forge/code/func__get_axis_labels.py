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
def _get_axis_labels(self, dimensions, xlabel=None, ylabel=None, zlabel=None):
    if self.xlabel is not None:
        xlabel = self.xlabel
    elif dimensions and xlabel is None:
        xdims = dimensions[0]
        xlabel = dim_axis_label(xdims) if xdims else ''
    if self.ylabel is not None:
        ylabel = self.ylabel
    elif len(dimensions) >= 2 and ylabel is None:
        ydims = dimensions[1]
        ylabel = dim_axis_label(ydims) if ydims else ''
    if getattr(self, 'zlabel', None) is not None:
        zlabel = self.zlabel
    elif isinstance(self.projection, str) and self.projection == '3d' and (len(dimensions) >= 3) and (zlabel is None):
        zlabel = dim_axis_label(dimensions[2]) if dimensions[2] else ''
    return (xlabel, ylabel, zlabel)