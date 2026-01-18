import numpy as np
from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.element.comparison import ComparisonTestCase
class BoxTests(ComparisonTestCase):

    def setUp(self):
        self.rotated_square = np.array([[-0.27059805, -0.65328148], [-0.65328148, 0.27059805], [0.27059805, 0.65328148], [0.65328148, -0.27059805], [-0.27059805, -0.65328148]])
        self.rotated_rect = np.array([[-0.73253782, -0.8446232], [-1.11522125, 0.07925633], [0.73253782, 0.8446232], [1.11522125, -0.07925633], [-0.73253782, -0.8446232]])

    def test_box_simple_constructor_rotated(self):
        box = Box(0, 0, 1, orientation=np.pi / 8)
        self.assertEqual(np.allclose(box.data[0], self.rotated_square), True)

    def test_box_tuple_constructor_rotated(self):
        box = Box(0, 0, (2, 1), orientation=np.pi / 8)
        self.assertEqual(np.allclose(box.data[0], self.rotated_rect), True)

    def test_box_aspect_constructor_rotated(self):
        box = Box(0, 0, 1, aspect=2, orientation=np.pi / 8)
        self.assertEqual(np.allclose(box.data[0], self.rotated_rect), True)