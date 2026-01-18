import math
import numpy as np
import pytest
from shapely import (
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry, EmptyGeometry
class TestEmptiness:

    def test_empty_class(self):
        with pytest.warns(FutureWarning):
            g = EmptyGeometry()
        assert g.is_empty

    def test_empty_base(self):
        with pytest.warns(FutureWarning):
            g = BaseGeometry()
        assert g.is_empty

    def test_empty_point(self):
        assert Point().is_empty

    def test_empty_multipoint(self):
        assert MultiPoint().is_empty

    def test_empty_geometry_collection(self):
        assert GeometryCollection().is_empty

    def test_empty_linestring(self):
        assert LineString().is_empty
        assert LineString(None).is_empty
        assert LineString([]).is_empty
        assert LineString(empty_generator()).is_empty

    def test_empty_multilinestring(self):
        assert MultiLineString([]).is_empty

    def test_empty_polygon(self):
        assert Polygon().is_empty
        assert Polygon(None).is_empty
        assert Polygon([]).is_empty
        assert Polygon(empty_generator()).is_empty

    def test_empty_multipolygon(self):
        assert MultiPolygon([]).is_empty

    def test_empty_linear_ring(self):
        assert LinearRing().is_empty
        assert LinearRing(None).is_empty
        assert LinearRing([]).is_empty
        assert LinearRing(empty_generator()).is_empty