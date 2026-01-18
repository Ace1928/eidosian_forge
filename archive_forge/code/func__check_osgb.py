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