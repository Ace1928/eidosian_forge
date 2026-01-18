import itertools
import types
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import groupby
from numbers import Number
from types import FunctionType
import numpy as np
import param
from ..streams import Params, Stream, streams_list_from_dict
from . import traversal, util
from .accessors import Opts, Redim
from .dimension import Dimension, ViewableElement
from .layout import AdjointLayout, Empty, Layout, Layoutable, NdLayout
from .ndmapping import NdMapping, UniformNdMapping, item_check
from .options import Store, StoreOptions
from .overlay import CompositeOverlay, NdOverlay, Overlay, Overlayable
def _dynamic_mul(self, dimensions, other, keys):
    """
        Implements dynamic version of overlaying operation overlaying
        DynamicMaps and HoloMaps where the key dimensions of one is
        a strict superset of the other.
        """
    if not isinstance(self, DynamicMap) or not isinstance(other, DynamicMap):
        keys = sorted(((d, v) for k in keys for d, v in k))
        grouped = {g: [v for _, v in group] for g, group in groupby(keys, lambda x: x[0])}
        dimensions = [d.clone(values=grouped[d.name]) for d in dimensions]
        map_obj = None
    map_obj = self if isinstance(self, DynamicMap) else other
    if isinstance(self, DynamicMap) and isinstance(other, DynamicMap):
        self_streams = util.dimensioned_streams(self)
        other_streams = util.dimensioned_streams(other)
        streams = list(util.unique_iterator(self_streams + other_streams))
    else:
        streams = map_obj.streams

    def dynamic_mul(*key, **kwargs):
        key_map = {d.name: k for d, k in zip(dimensions, key)}
        layers = []
        try:
            self_el = self.select(HoloMap, **key_map) if self.kdims else self[()]
            layers.append(self_el)
        except KeyError:
            pass
        try:
            other_el = other.select(HoloMap, **key_map) if other.kdims else other[()]
            layers.append(other_el)
        except KeyError:
            pass
        return Overlay(layers)
    callback = Callable(dynamic_mul, inputs=[self, other])
    callback._is_overlay = True
    if map_obj:
        return map_obj.clone(callback=callback, shared_data=False, kdims=dimensions, streams=streams)
    else:
        return DynamicMap(callback=callback, kdims=dimensions, streams=streams)