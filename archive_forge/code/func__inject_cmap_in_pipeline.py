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
@classmethod
def _inject_cmap_in_pipeline(cls, pipeline, cmap):
    operations = []
    for op in pipeline.operations:
        if hasattr(op, 'cmap'):
            op = op.instance(cmap=cmap)
        operations.append(op)
    return pipeline.instance(operations=operations)