import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
class UniformNdMappingTest(ComparisonTestCase):

    def test_collapse_nested(self):
        inner1 = UniformNdMapping({1: Dataset([(1, 2)], ['x', 'y'])}, 'Y')
        inner2 = UniformNdMapping({1: Dataset([(3, 4)], ['x', 'y'])}, 'Y')
        outer = UniformNdMapping({1: inner1, 2: inner2}, 'X')
        collapsed = outer.collapse()
        expected = Dataset([(1, 1, 1, 2), (2, 1, 3, 4)], ['X', 'Y', 'x', 'y'])
        self.assertEqual(collapsed, expected)