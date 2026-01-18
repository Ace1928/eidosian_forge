import random
import unittest
from functools import partial
from itertools import islice
import pytest
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import MultiPolygon, Point
from shapely.ops import cascaded_union, unary_union
class UnionTestCase(unittest.TestCase):

    def test_cascaded_union(self):
        r = partial(random.uniform, -20.0, 20.0)
        points = [Point(r(), r()) for i in range(100)]
        spots = [p.buffer(2.5) for p in points]
        with pytest.warns(ShapelyDeprecationWarning, match='is deprecated'):
            u = cascaded_union(spots)
        assert u.geom_type in ('Polygon', 'MultiPolygon')

    def setUp(self):
        self.coords = zip(list(islice(halton(5), 20, 120)), list(islice(halton(7), 20, 120)))

    def test_unary_union(self):
        patches = [Point(xy).buffer(0.05) for xy in self.coords]
        u = unary_union(patches)
        assert u.geom_type == 'MultiPolygon'
        assert u.area == pytest.approx(0.718572540569)

    def test_unary_union_multi(self):
        patches = MultiPolygon([Point(xy).buffer(0.05) for xy in self.coords])
        assert unary_union(patches).area == pytest.approx(0.71857254056)
        assert unary_union([patches, patches]).area == pytest.approx(0.71857254056)