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
class GenericCompositePlot(DimensionedPlot):

    def __init__(self, layout, keys=None, dimensions=None, **params):
        if 'uniform' not in params:
            params['uniform'] = traversal.uniform(layout)
        self.top_level = keys is None
        if self.top_level:
            dimensions, keys = traversal.unique_dimkeys(layout)
        dynamic, unbounded = get_dynamic_mode(layout)
        if unbounded:
            initialize_unbounded(layout, dimensions, keys[0])
        self.layout = layout
        super().__init__(keys=keys, dynamic=dynamic, dimensions=dimensions, **params)
        nested_streams = layout.traverse(lambda x: get_nested_streams(x), [DynamicMap])
        self.streams = list({s for streams in nested_streams for s in streams})
        self._link_dimensioned_streams()

    def _link_dimensioned_streams(self):
        """
        Should perform any linking required to update titles when dimensioned
        streams change.
        """

    def _get_frame(self, key):
        """
        Creates a clone of the Layout with the nth-frame for each
        Element.
        """
        cached = self.current_key is None
        layout_frame = self.layout.clone(shared_data=False)
        if key == self.current_key and (not self._force):
            return self.current_frame
        else:
            self.current_key = key
        key_map = dict(zip([d.name for d in self.dimensions], key))
        for path, item in self.layout.items():
            frame = get_nested_plot_frame(item, key_map, cached)
            if frame is not None:
                layout_frame[path] = frame
        traverse_setter(self, '_force', False)
        self.current_frame = layout_frame
        return layout_frame

    def _format_title_components(self, key, dimensions=True, separator='\n'):
        dim_title = self._frame_title(key, 3, separator) if dimensions else ''
        layout = self.layout
        type_name = type(self.layout).__name__
        group = util.bytes_to_unicode(layout.group if layout.group != type_name else '')
        label = util.bytes_to_unicode(layout.label)
        return (label, group, type_name, dim_title)