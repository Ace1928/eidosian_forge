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
class _SelectionExprLayers(Derived):
    exprs = param.List(constant=True)

    def __init__(self, expr_override, cross_filter_set, **params):
        super().__init__([expr_override, cross_filter_set], exclusive=True, **params)

    @classmethod
    def transform_function(cls, stream_values, constants):
        override_expr_values = stream_values[0]
        cross_filter_set_values = stream_values[1]
        if override_expr_values.get('selection_expr', None) is not None:
            return {'exprs': [True, override_expr_values['selection_expr']]}
        else:
            return {'exprs': [True, cross_filter_set_values['selection_expr']]}