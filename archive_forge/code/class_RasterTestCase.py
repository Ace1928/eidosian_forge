import numpy as np
from holoviews import Image
from holoviews.core import BoundingBox, Dimension
from holoviews.core.element import HoloMap
from holoviews.element.comparison import ComparisonTestCase
class RasterTestCase(ComparisonTestCase):

    def setUp(self):
        self.arr1 = np.array([[1, 2], [3, 4]])
        self.arr2 = np.array([[10, 2], [3, 4]])
        self.arr3 = np.array([[10, 2], [3, 40]])
        self.mat1 = Image(self.arr1, bounds=BoundingBox())
        self.mat2 = Image(self.arr2, bounds=BoundingBox())
        self.mat3 = Image(self.arr3, bounds=BoundingBox())
        self.mat4 = Image(self.arr1, bounds=BoundingBox(radius=0.3))
        self.mat5 = Image(self.arr2, bounds=BoundingBox(radius=0.3))