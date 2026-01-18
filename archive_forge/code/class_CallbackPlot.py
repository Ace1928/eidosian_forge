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
class CallbackPlot:
    backend = None

    def _construct_callbacks(self):
        """
        Initializes any callbacks for streams which have defined
        the plotted object as a source.
        """
        source_streams = []
        cb_classes = set()
        registry = list(Stream.registry.items())
        callbacks = Stream._callbacks[self.backend]
        for source in self.link_sources:
            streams = [s for src, streams in registry for s in streams if src is source or (src._plot_id is not None and src._plot_id == source._plot_id)]
            cb_classes |= {(callbacks[type(stream)], stream) for stream in streams if type(stream) in callbacks and stream.linked and (stream.source is not None)}
        cbs = []
        sorted_cbs = sorted(cb_classes, key=lambda x: id(x[0]))
        for cb, group in groupby(sorted_cbs, lambda x: x[0]):
            cb_streams = [s for _, s in group]
            for cb_stream in cb_streams:
                if cb_stream not in source_streams:
                    source_streams.append(cb_stream)
            cbs.append(cb(self, cb_streams, source))
        return (cbs, source_streams)

    @property
    def link_sources(self):
        """Returns potential Link or Stream sources."""
        if isinstance(self, GenericOverlayPlot):
            zorders = []
        elif self.batched:
            zorders = list(range(self.zorder, self.zorder + len(self.hmap.last)))
        else:
            zorders = [self.zorder]
        if isinstance(self, GenericOverlayPlot) and (not self.batched):
            sources = [self.hmap.last]
        elif not self.static or isinstance(self.hmap, DynamicMap):
            sources = [o for i, inputs in self.stream_sources.items() for o in inputs if i in zorders]
        else:
            sources = [self.hmap.last]
        return sources