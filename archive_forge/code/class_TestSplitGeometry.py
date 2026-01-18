import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import (
from shapely.ops import linemerge, split, unary_union
class TestSplitGeometry(unittest.TestCase):

    def helper(self, geom, splitter, expected_chunks):
        s = split(geom, splitter)
        assert s.geom_type == 'GeometryCollection'
        assert len(s.geoms) == expected_chunks
        if expected_chunks > 1:
            if s.geoms[0].geom_type == 'LineString':
                self.assertTrue(linemerge(s).simplify(1e-06).equals(geom))
            elif s.geoms[0].geom_type == 'Polygon':
                union = unary_union(s).simplify(1e-06)
                assert union.equals(geom)
                assert union.area == geom.area
            else:
                raise ValueError
        elif expected_chunks == 1:
            assert s.geoms[0].equals(geom)

    def test_split_closed_line_with_point(self):
        ls = LineString([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
        splitter = Point(0, 0)
        self.helper(ls, splitter, 1)