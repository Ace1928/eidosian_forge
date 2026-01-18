import numpy as np
import shapely
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame, GeoSeries, clip
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
import pytest
from geopandas.tools.clip import _mask_is_list_like_rectangle
@pytest.mark.parametrize('mask_fixture_name', mask_variants_single_rectangle)
class TestClipWithSingleRectangleGdf:

    @pytest.fixture
    def mask(self, mask_fixture_name, request):
        return request.getfixturevalue(mask_fixture_name)

    def test_returns_gdf(self, point_gdf, mask):
        """Test that function returns a GeoDataFrame (or GDF-like) object."""
        out = clip(point_gdf, mask)
        assert isinstance(out, GeoDataFrame)

    def test_returns_series(self, point_gdf, mask):
        """Test that function returns a GeoSeries if GeoSeries is passed."""
        out = clip(point_gdf.geometry, mask)
        assert isinstance(out, GeoSeries)

    def test_clip_points(self, point_gdf, mask):
        """Test clipping a points GDF with a generic polygon geometry."""
        clip_pts = clip(point_gdf, mask)
        pts = np.array([[2, 2], [3, 4], [9, 8]])
        exp = GeoDataFrame([Point(xy) for xy in pts], columns=['geometry'], crs='EPSG:3857')
        assert_geodataframe_equal(clip_pts, exp)

    def test_clip_points_geom_col_rename(self, point_gdf, mask):
        """Test clipping a points GDF with a generic polygon geometry."""
        point_gdf_geom_col_rename = point_gdf.rename_geometry('geometry2')
        clip_pts = clip(point_gdf_geom_col_rename, mask)
        pts = np.array([[2, 2], [3, 4], [9, 8]])
        exp = GeoDataFrame([Point(xy) for xy in pts], columns=['geometry2'], crs='EPSG:3857', geometry='geometry2')
        assert_geodataframe_equal(clip_pts, exp)

    def test_clip_poly(self, buffered_locations, mask):
        """Test clipping a polygon GDF with a generic polygon geometry."""
        clipped_poly = clip(buffered_locations, mask)
        assert len(clipped_poly.geometry) == 3
        assert all(clipped_poly.geom_type == 'Polygon')

    def test_clip_poly_geom_col_rename(self, buffered_locations, mask):
        """Test clipping a polygon GDF with a generic polygon geometry."""
        poly_gdf_geom_col_rename = buffered_locations.rename_geometry('geometry2')
        clipped_poly = clip(poly_gdf_geom_col_rename, mask)
        assert len(clipped_poly.geometry) == 3
        assert 'geometry' not in clipped_poly.keys()
        assert 'geometry2' in clipped_poly.keys()

    def test_clip_poly_series(self, buffered_locations, mask):
        """Test clipping a polygon GDF with a generic polygon geometry."""
        clipped_poly = clip(buffered_locations.geometry, mask)
        assert len(clipped_poly) == 3
        assert all(clipped_poly.geom_type == 'Polygon')

    def test_clip_multipoly_keep_geom_type(self, multi_poly_gdf, mask):
        """Test a multi poly object where the return includes a sliver.
        Also the bounds of the object should == the bounds of the clip object
        if they fully overlap (as they do in these fixtures)."""
        clipped = clip(multi_poly_gdf, mask, keep_geom_type=True)
        expected_bounds = mask if _mask_is_list_like_rectangle(mask) else mask.total_bounds
        assert np.array_equal(clipped.total_bounds, expected_bounds)
        assert clipped.geom_type.isin(['Polygon', 'MultiPolygon']).all()

    def test_clip_multiline(self, multi_line, mask):
        """Test that clipping a multiline feature with a poly returns expected
        output."""
        clipped = clip(multi_line, mask)
        assert clipped.geom_type[0] == 'MultiLineString'

    def test_clip_multipoint(self, multi_point, mask):
        """Clipping a multipoint feature with a polygon works as expected.
        should return a geodataframe with a single multi point feature"""
        clipped = clip(multi_point, mask)
        assert clipped.geom_type[0] == 'MultiPoint'
        assert hasattr(clipped, 'attr')
        assert len(clipped) == 2
        clipped_mutltipoint = MultiPoint([Point(2, 2), Point(3, 4), Point(9, 8)])
        assert clipped.iloc[0].geometry.wkt == clipped_mutltipoint.wkt
        shape_for_points = box(*mask) if _mask_is_list_like_rectangle(mask) else mask.unary_union
        assert all(clipped.intersects(shape_for_points))

    def test_clip_lines(self, two_line_gdf, mask):
        """Test what happens when you give the clip_extent a line GDF."""
        clip_line = clip(two_line_gdf, mask)
        assert len(clip_line.geometry) == 2

    def test_mixed_geom(self, mixed_gdf, mask):
        """Test clipping a mixed GeoDataFrame"""
        clipped = clip(mixed_gdf, mask)
        assert clipped.geom_type[0] == 'Point' and clipped.geom_type[1] == 'Polygon' and (clipped.geom_type[2] == 'LineString')

    def test_mixed_series(self, mixed_gdf, mask):
        """Test clipping a mixed GeoSeries"""
        clipped = clip(mixed_gdf.geometry, mask)
        assert clipped.geom_type[0] == 'Point' and clipped.geom_type[1] == 'Polygon' and (clipped.geom_type[2] == 'LineString')

    def test_clip_with_line_extra_geom(self, sliver_line, mask):
        """When the output of a clipped line returns a geom collection,
        and keep_geom_type is True, no geometry collections should be returned."""
        clipped = clip(sliver_line, mask, keep_geom_type=True)
        assert len(clipped.geometry) == 1
        assert not (clipped.geom_type == 'GeometryCollection').any()

    def test_clip_no_box_overlap(self, pointsoutside_nooverlap_gdf, mask):
        """Test clip when intersection is empty and boxes do not overlap."""
        clipped = clip(pointsoutside_nooverlap_gdf, mask)
        assert len(clipped) == 0

    def test_clip_box_overlap(self, pointsoutside_overlap_gdf, mask):
        """Test clip when intersection is empty and boxes do overlap."""
        clipped = clip(pointsoutside_overlap_gdf, mask)
        assert len(clipped) == 0

    def test_warning_extra_geoms_mixed(self, mixed_gdf, mask):
        """Test the correct warnings are raised if keep_geom_type is
        called on a mixed GDF"""
        with pytest.warns(UserWarning):
            clip(mixed_gdf, mask, keep_geom_type=True)

    def test_warning_geomcoll(self, geomcol_gdf, mask):
        """Test the correct warnings are raised if keep_geom_type is
        called on a GDF with GeometryCollection"""
        with pytest.warns(UserWarning):
            clip(geomcol_gdf, mask, keep_geom_type=True)