import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
class TestInitializeDynamic(ComparisonTestCase):

    def test_dynamicmap_default_initializes(self):
        dims = [Dimension('N', default=5, range=(0, 10))]
        dmap = DynamicMap(lambda N: Curve([1, N, 5]), kdims=dims)
        initialize_dynamic(dmap)
        self.assertEqual(dmap.keys(), [5])

    def test_dynamicmap_numeric_values_initializes(self):
        dims = [Dimension('N', values=[10, 5, 0])]
        dmap = DynamicMap(lambda N: Curve([1, N, 5]), kdims=dims)
        initialize_dynamic(dmap)
        self.assertEqual(dmap.keys(), [0])