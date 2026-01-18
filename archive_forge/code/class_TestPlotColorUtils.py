import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
class TestPlotColorUtils(ComparisonTestCase):

    def test_process_cmap_list_cycle(self):
        colors = process_cmap(['#ffffff', '#959595', '#000000'], 4)
        self.assertEqual(colors, ['#ffffff', '#959595', '#000000', '#ffffff'])

    def test_process_cmap_cycle(self):
        colors = process_cmap(Cycle(values=['#ffffff', '#959595', '#000000']), 4)
        self.assertEqual(colors, ['#ffffff', '#959595', '#000000', '#ffffff'])

    def test_process_cmap_invalid_str(self):
        with self.assertRaises(ValueError):
            process_cmap('NonexistentColorMap', 3)

    def test_process_cmap_invalid_type(self):
        with self.assertRaises(TypeError):
            process_cmap({'A', 'B', 'C'}, 3)