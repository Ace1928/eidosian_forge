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
class GenericLayoutPlot(GenericCompositePlot):
    """
    A GenericLayoutPlot accepts either a Layout or a NdLayout and
    displays the elements in a cartesian grid in scanline order.
    """
    transpose = param.Boolean(default=False, doc='\n        Whether to transpose the layout when plotting. Switches\n        from row-based left-to-right and top-to-bottom scanline order\n        to column-based top-to-bottom and left-to-right order.')

    def __init__(self, layout, **params):
        if not isinstance(layout, (NdLayout, Layout)):
            raise ValueError('GenericLayoutPlot only accepts Layout objects.')
        if len(layout.values()) == 0:
            raise SkipRendering(warn=False)
        super().__init__(layout, **params)
        self.subplots = {}
        self.rows, self.cols = layout.shape[::-1] if self.transpose else layout.shape
        self.coords = list(product(range(self.rows), range(self.cols)))