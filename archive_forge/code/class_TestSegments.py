import numpy as np
from holoviews.element.comparison import ComparisonTestCase
from shapely.geometry import (
from geoviews.element import Rectangles, Path, Polygons, Points, Segments
class TestSegments(ComparisonTestCase):

    def test_empty_geom_conversion(self):
        segs = Segments([])
        self.assertEqual(segs.geom(), GeometryCollection())

    def test_single_geom_conversion(self):
        segs = Segments([(0, 0, 1, 1)])
        geom = segs.geom()
        self.assertIsInstance(geom, LineString)
        self.assertEqual(np.array(geom.coords), np.array([[0, 0], [1, 1]]))

    def test_multi_geom_conversion(self):
        segs = Segments([(0, 0, 1, 1), (1.5, 2, 3, 1)])
        geom = segs.geom()
        self.assertIsInstance(geom, MultiLineString)
        self.assertEqual(len(geom.geoms), 2)
        self.assertEqual(np.array(geom.geoms[0].coords), np.array([[0, 0], [1, 1]]))
        self.assertEqual(np.array(geom.geoms[1].coords), np.array([[1.5, 2], [3, 1]]))