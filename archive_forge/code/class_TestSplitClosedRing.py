import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import (
from shapely.ops import linemerge, split, unary_union
class TestSplitClosedRing(TestSplitGeometry):
    ls = LineString([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])

    def test_split_closed_ring_with_point(self):
        splitter = Point([0.0, 0.0])
        self.helper(self.ls, splitter, 1)
        splitter = Point([0.0, 0.5])
        self.helper(self.ls, splitter, 2)
        result = split(self.ls, splitter)
        assert result.geoms[0].coords[:] == [(0, 0), (0.0, 0.5)]
        assert result.geoms[1].coords[:] == [(0.0, 0.5), (0, 1), (1, 1), (1, 0), (0, 0)]
        splitter = Point([0.5, 0.0])
        self.helper(self.ls, splitter, 2)
        result = split(self.ls, splitter)
        assert result.geoms[0].coords[:] == [(0, 0), (0, 1), (1, 1), (1, 0), (0.5, 0)]
        assert result.geoms[1].coords[:] == [(0.5, 0), (0, 0)]
        splitter = Point([2.0, 2.0])
        self.helper(self.ls, splitter, 1)