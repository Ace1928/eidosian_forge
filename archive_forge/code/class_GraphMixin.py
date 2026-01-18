import numpy as np
from ..core import Dataset, Dimension, util
from ..element import Bars, Graph
from ..element.util import categorical_aggregate2d
from .util import get_axis_padding
class GraphMixin:

    def _get_axis_dims(self, element):
        if isinstance(element, Graph):
            element = element.nodes
        return element.dimensions()[:2]

    def get_extents(self, element, ranges, range_type='combined', **kwargs):
        return super().get_extents(element.nodes, ranges, range_type)