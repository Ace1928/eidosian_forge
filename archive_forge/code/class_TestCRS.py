import copy
from io import BytesIO
import os
from pathlib import Path
import pickle
import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from numpy.testing import assert_array_almost_equal as assert_arr_almost_eq
import pyproj
import pytest
import shapely.geometry as sgeom
import cartopy.crs as ccrs
class TestCRS:

    def test_hash(self):
        stereo = ccrs.Stereographic(90)
        north = ccrs.NorthPolarStereo()
        assert stereo == north
        assert not stereo != north
        assert hash(stereo) == hash(north)
        assert ccrs.Geodetic() == ccrs.Geodetic()

    @pytest.mark.parametrize('approx', [True, False])
    def test_osni(self, approx):
        osni = ccrs.OSNI(approx=approx)
        ll = ccrs.Geodetic()
        lat, lon = np.array([54.5622169298669, -5.54159863617957], dtype=np.double)
        east, north = np.array([359000, 371000], dtype=np.double)
        assert_arr_almost_eq(osni.transform_point(lon, lat, ll), np.array([east, north]), -1)
        assert_arr_almost_eq(ll.transform_point(east, north, osni), np.array([lon, lat]), 3)

    def _check_osgb(self, osgb):
        precision = 1
        if os.environ.get('PROJ_NETWORK') != 'ON':
            grid_name = 'uk_os_OSTN15_NTv2_OSGBtoETRS.tif'
            available = Path(pyproj.datadir.get_data_dir(), grid_name).exists() or Path(pyproj.datadir.get_user_data_dir(), grid_name).exists()
            if not available:
                import warnings
                warnings.warn(f'{grid_name} is unavailable; testing OSGB at reduced precision')
                precision = -1
        ll = ccrs.Geodetic()
        lat, lon = np.array([50.462023, -3.478831], dtype=np.double)
        east, north = np.array([295132.1, 63512.6], dtype=np.double)
        assert_almost_equal(osgb.transform_point(lon, lat, ll), [east, north], decimal=precision)
        assert_almost_equal(ll.transform_point(east, north, osgb), [lon, lat], decimal=2)
        r_lon, r_lat = ll.transform_point(east, north, osgb)
        r_inverted = np.array(osgb.transform_point(r_lon, r_lat, ll))
        assert_arr_almost_eq(r_inverted, [east, north], 3)
        r_east, r_north = osgb.transform_point(lon, lat, ll)
        r_inverted = np.array(ll.transform_point(r_east, r_north, osgb))
        assert_arr_almost_eq(r_inverted, [lon, lat])

    @pytest.mark.parametrize('approx', [True, False])
    def test_osgb(self, approx):
        self._check_osgb(ccrs.OSGB(approx=approx))

    def test_epsg(self):
        uk = ccrs.epsg(27700)
        assert uk.epsg_code == 27700
        expected_x = (-104009.357, 688806.007)
        expected_y = (-8908.37, 1256558.45)
        expected_threshold = 7928.15
        if pyproj.__proj_version__ >= '9.2.0':
            expected_x = (-104728.764, 688806.007)
            expected_y = (-8908.36, 1256616.32)
            expected_threshold = 7935.34
        assert_almost_equal(uk.x_limits, expected_x, decimal=3)
        assert_almost_equal(uk.y_limits, expected_y, decimal=2)
        assert_almost_equal(uk.threshold, expected_threshold, decimal=2)
        self._check_osgb(uk)

    def test_epsg_compound_crs(self):
        projection = ccrs.epsg(5973)
        assert projection.epsg_code == 5973

    def test_europp(self):
        europp = ccrs.EuroPP()
        proj4_init = europp.proj4_init
        assert '+proj=utm' in proj4_init
        assert '+zone=32' in proj4_init
        assert '+ellps=intl' in proj4_init

    def test_transform_points_nD(self):
        rlons = np.array([[350.0, 352.0, 354.0], [350.0, 352.0, 354.0]])
        rlats = np.array([[-5.0, -0.0, 1.0], [-4.0, -1.0, 0.0]])
        src_proj = ccrs.RotatedGeodetic(pole_longitude=178.0, pole_latitude=38.0)
        target_proj = ccrs.Geodetic()
        res = target_proj.transform_points(x=rlons, y=rlats, src_crs=src_proj)
        unrotated_lon = res[..., 0]
        unrotated_lat = res[..., 1]
        solx = np.array([[-16.42176094, -14.85892262, -11.9062752], [-16.71055023, -14.58434624, -11.68799988]])
        soly = np.array([[46.00724251, 51.29188893, 52.59101488], [46.98728486, 50.30706042, 51.60004528]])
        assert_arr_almost_eq(unrotated_lon, solx)
        assert_arr_almost_eq(unrotated_lat, soly)

    def test_transform_points_1D(self):
        rlons = np.array([350.0, 352.0, 354.0, 356.0])
        rlats = np.array([-5.0, -0.0, 5.0, 10.0])
        src_proj = ccrs.RotatedGeodetic(pole_longitude=178.0, pole_latitude=38.0)
        target_proj = ccrs.Geodetic()
        res = target_proj.transform_points(x=rlons, y=rlats, src_crs=src_proj)
        unrotated_lon = res[..., 0]
        unrotated_lat = res[..., 1]
        solx = np.array([-16.42176094, -14.85892262, -12.88946157, -10.35078336])
        soly = np.array([46.00724251, 51.29188893, 56.55031485, 61.77015703])
        assert_arr_almost_eq(unrotated_lon, solx)
        assert_arr_almost_eq(unrotated_lat, soly)

    def test_transform_points_xyz(self):
        rx = np.array([2574325.16])
        ry = np.array([837562.0])
        rz = np.array([5761325.0])
        src_proj = ccrs.Geocentric()
        target_proj = ccrs.Geodetic()
        res = target_proj.transform_points(x=rx, y=ry, z=rz, src_crs=src_proj)
        glat = res[..., 0]
        glon = res[..., 1]
        galt = res[..., 2]
        solx = np.array([18.0224043189])
        soly = np.array([64.9796515089])
        solz = np.array([5048.03893734])
        assert_arr_almost_eq(glat, solx)
        assert_arr_almost_eq(glon, soly)
        assert_arr_almost_eq(galt, solz)

    def test_transform_points_180(self):
        x = np.array([-190, 190])
        y = np.array([0, 0])
        proj = ccrs.PlateCarree()
        res = proj.transform_points(x=x, y=y, src_crs=proj)
        assert_array_equal(res[..., :2], [[170, 0], [-170, 0]])

    def test_globe(self):
        rugby_globe = ccrs.Globe(semimajor_axis=9000000, semiminor_axis=9000000, ellipse=None)
        footy_globe = ccrs.Globe(semimajor_axis=1000000, semiminor_axis=1000000, ellipse=None)
        rugby_moll = ccrs.Mollweide(globe=rugby_globe)
        footy_moll = ccrs.Mollweide(globe=footy_globe)
        rugby_pt = rugby_moll.transform_point(10, 10, rugby_moll.as_geodetic())
        footy_pt = footy_moll.transform_point(10, 10, footy_moll.as_geodetic())
        assert_arr_almost_eq(rugby_pt, (1400915, 1741319), decimal=0)
        assert_arr_almost_eq(footy_pt, (155657, 193479), decimal=0)

    def test_project_point(self):
        point = sgeom.Point([0, 45])
        multi_point = sgeom.MultiPoint([point, sgeom.Point([180, 45])])
        pc = ccrs.PlateCarree()
        pc_rotated = ccrs.PlateCarree(central_longitude=180)
        result = pc_rotated.project_geometry(point, pc)
        assert_arr_almost_eq(result.xy, [[-180.0], [45.0]])
        result = pc_rotated.project_geometry(multi_point, pc)
        assert isinstance(result, sgeom.MultiPoint)
        assert len(result.geoms) == 2
        assert_arr_almost_eq(result.geoms[0].xy, [[-180.0], [45.0]])
        assert_arr_almost_eq(result.geoms[1].xy, [[0], [45.0]])

    def test_utm(self):
        utm30n = ccrs.UTM(30)
        ll = ccrs.Geodetic()
        lat, lon = np.array([51.5, -3.0], dtype=np.double)
        east, north = np.array([500000, 5705429.2], dtype=np.double)
        assert_arr_almost_eq(utm30n.transform_point(lon, lat, ll), [east, north], decimal=1)
        assert_arr_almost_eq(ll.transform_point(east, north, utm30n), [lon, lat], decimal=1)
        utm38s = ccrs.UTM(38, southern_hemisphere=True)
        lat, lon = np.array([-18.92, 47.5], dtype=np.double)
        east, north = np.array([763316.7, 7906160.8], dtype=np.double)
        assert_arr_almost_eq(utm38s.transform_point(lon, lat, ll), [east, north], decimal=1)
        assert_arr_almost_eq(ll.transform_point(east, north, utm38s), [lon, lat], decimal=1)