import json
import unittest
import pytest
from shapely.errors import GeometryTypeError
from shapely.geometry import LineString, Point, shape
from shapely.ops import substring
class SubstringTestCase(unittest.TestCase):

    def setUp(self):
        self.point = Point(1, 1)
        self.line1 = LineString([(0, 0), (2, 0)])
        self.line2 = LineString([(3, 0), (3, 6), (4.5, 6)])
        self.line3 = LineString(((0, i) for i in range(5)))

    def test_return_startpoint(self):
        assert substring(self.line1, -500, -600).equals(Point(0, 0))
        assert substring(self.line1, -500, -500).equals(Point(0, 0))
        assert substring(self.line1, -1, -1.1, True).equals(Point(0, 0))
        assert substring(self.line1, -1.1, -1.1, True).equals(Point(0, 0))

    def test_return_endpoint(self):
        assert substring(self.line1, 500, 600).equals(Point(2, 0))
        assert substring(self.line1, 500, 500).equals(Point(2, 0))
        assert substring(self.line1, 1, 1.1, True).equals(Point(2, 0))
        assert substring(self.line1, 1.1, 1.1, True).equals(Point(2, 0))

    def test_return_midpoint(self):
        assert substring(self.line1, 0.5, 0.5).equals(Point(0.5, 0))
        assert substring(self.line1, -0.5, -0.5).equals(Point(1.5, 0))
        assert substring(self.line1, 0.5, 0.5, True).equals(Point(1, 0))
        assert substring(self.line1, -0.5, -0.5, True).equals(Point(1, 0))
        assert substring(self.line1, 1.5, -0.5).equals(Point(1.5, 0))
        assert substring(self.line1, -0.5, 1.5).equals(Point(1.5, 0))
        assert substring(self.line1, -0.7, 0.3, True).equals(Point(0.6, 0))
        assert substring(self.line1, 0.3, -0.7, True).equals(Point(0.6, 0))

    def test_return_startsubstring(self):
        assert substring(self.line1, -500, 0.6).wkt == LineString([(0, 0), (0.6, 0)]).wkt
        assert substring(self.line1, -1.1, 0.6, True).wkt == LineString([(0, 0), (1.2, 0)]).wkt

    def test_return_startsubstring_reversed(self):
        assert substring(self.line1, -1, -500).wkt == LineString([(1, 0), (0, 0)]).wkt
        assert substring(self.line3, 3.5, 0).wkt == LineString([(0, 3.5), (0, 3), (0, 2), (0, 1), (0, 0)]).wkt
        assert substring(self.line3, -1.5, -500).wkt == LineString([(0, 2.5), (0, 2), (0, 1), (0, 0)]).wkt
        assert substring(self.line1, -0.5, -1.1, True).wkt == LineString([(1.0, 0), (0, 0)]).wkt
        assert substring(self.line3, 0.5, 0, True).wkt == LineString([(0, 2.0), (0, 1), (0, 0)]).wkt
        assert substring(self.line3, -0.5, -1.1, True).wkt == LineString([(0, 2.0), (0, 1), (0, 0)]).wkt

    def test_return_endsubstring(self):
        assert substring(self.line1, 0.6, 500).wkt == LineString([(0.6, 0), (2, 0)]).wkt
        assert substring(self.line1, 0.6, 1.1, True).wkt == LineString([(1.2, 0), (2, 0)]).wkt

    def test_return_endsubstring_reversed(self):
        assert substring(self.line1, 500, -1).wkt == LineString([(2, 0), (1, 0)]).wkt
        assert substring(self.line3, 4, 2.5).wkt == LineString([(0, 4), (0, 3), (0, 2.5)]).wkt
        assert substring(self.line3, 500, -1.5).wkt == LineString([(0, 4), (0, 3), (0, 2.5)]).wkt
        assert substring(self.line1, 1.1, -0.5, True).wkt == LineString([(2, 0), (1.0, 0)]).wkt
        assert substring(self.line3, 1, 0.5, True).wkt == LineString([(0, 4), (0, 3), (0, 2.0)]).wkt
        assert substring(self.line3, 1.1, -0.5, True).wkt == LineString([(0, 4), (0, 3), (0, 2.0)]).wkt

    def test_return_midsubstring(self):
        assert substring(self.line1, 0.5, 0.6).wkt == LineString([(0.5, 0), (0.6, 0)]).wkt
        assert substring(self.line1, -0.6, -0.5).wkt == LineString([(1.4, 0), (1.5, 0)]).wkt
        assert substring(self.line1, 0.5, 0.6, True).wkt == LineString([(1, 0), (1.2, 0)]).wkt
        assert substring(self.line1, -0.6, -0.5, True).wkt == LineString([(0.8, 0), (1, 0)]).wkt

    def test_return_midsubstring_reversed(self):
        assert substring(self.line1, 0.6, 0.5).wkt == LineString([(0.6, 0), (0.5, 0)]).wkt
        assert substring(self.line1, -0.5, -0.6).wkt == LineString([(1.5, 0), (1.4, 0)]).wkt
        assert substring(self.line1, 0.6, 0.5, True).wkt == LineString([(1.2, 0), (1, 0)]).wkt
        assert substring(self.line1, -0.5, -0.6, True).wkt == LineString([(1, 0), (0.8, 0)]).wkt
        assert substring(self.line3, 3.5, 2.5).wkt == LineString([(0, 3.5), (0, 3), (0, 2.5)]).wkt
        assert substring(self.line3, -0.5, -1.5).wkt == LineString([(0, 3.5), (0, 3), (0, 2.5)]).wkt
        assert substring(self.line3, 3.5, -1.5).wkt == LineString([(0, 3.5), (0, 3), (0, 2.5)]).wkt
        assert substring(self.line3, -0.5, 2.5).wkt == LineString([(0, 3.5), (0, 3), (0, 2.5)]).wkt
        assert substring(self.line3, 0.875, 0.625, True).wkt == LineString([(0, 3.5), (0, 3), (0, 2.5)]).wkt
        assert substring(self.line3, -0.125, -0.375, True).wkt == LineString([(0, 3.5), (0, 3), (0, 2.5)]).wkt
        assert substring(self.line3, 0.875, -0.375, True).wkt == LineString([(0, 3.5), (0, 3), (0, 2.5)]).wkt
        assert substring(self.line3, -0.125, 0.625, True).wkt == LineString([(0, 3.5), (0, 3), (0, 2.5)]).wkt

    def test_return_substring_with_vertices(self):
        assert substring(self.line2, 1, 7).wkt == LineString([(3, 1), (3, 6), (4, 6)]).wkt
        assert substring(self.line2, 0.2, 0.9, True).wkt == LineString([(3, 1.5), (3, 6), (3.75, 6)]).wkt
        assert substring(self.line2, 0, 0.9, True).wkt == LineString([(3, 0), (3, 6), (3.75, 6)]).wkt
        assert substring(self.line2, 0.2, 1, True).wkt == LineString([(3, 1.5), (3, 6), (4.5, 6)]).wkt

    def test_return_substring_issue682(self):
        assert list(substring(self.line2, 0.1, 0).coords) == [(3.0, 0.1), (3.0, 0.0)]

    def test_return_substring_issue848(self):
        line = shape(json.loads(data_issue_848))
        cut_line = substring(line, 0.7, 0.8, normalized=True)
        assert len(cut_line.coords) == 53

    def test_raise_type_error(self):
        with pytest.raises(GeometryTypeError):
            substring(Point(0, 0), 0, 0)

    def test_return_z_coord_issue1699(self):
        line_z = LineString([(0, 0, 0), (2, 0, 0)])
        assert substring(line_z, 0, 0.5, True).wkt == LineString([(0, 0, 0), (1, 0, 0)]).wkt
        assert substring(line_z, 0.5, 0, True).wkt == LineString([(1, 0, 0), (0, 0, 0)]).wkt