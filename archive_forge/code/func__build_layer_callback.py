from collections import namedtuple
import numpy as np
import param
from param.parameterized import bothmethod
from .core.data import Dataset
from .core.element import Element, Layout
from .core.layout import AdjointLayout
from .core.options import CallbackError, Store
from .core.overlay import NdOverlay, Overlay
from .core.spaces import GridSpace
from .streams import (
from .util import DynamicMap
def _build_layer_callback(self, element, exprs, layer_number, cmap, cache, **kwargs):
    selection = self._select(element, exprs[layer_number], cache)
    pipeline = element.pipeline
    if cmap is not None:
        pipeline = self._inject_cmap_in_pipeline(pipeline, cmap)
    if element is selection:
        return pipeline(element.dataset)
    else:
        return pipeline(selection)