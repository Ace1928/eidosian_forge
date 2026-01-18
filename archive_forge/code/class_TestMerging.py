import warnings
import pandas as pd
import pyproj
import pytest
from geopandas._compat import PANDAS_GE_21
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_index_equal
from shapely.geometry import Point
from geopandas import GeoDataFrame, GeoSeries
class TestMerging:

    def setup_method(self):
        self.gseries = GeoSeries([Point(i, i) for i in range(3)])
        self.series = pd.Series([1, 2, 3])
        self.gdf = GeoDataFrame({'geometry': self.gseries, 'values': range(3)})
        self.df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [0.1, 0.2, 0.3]})

    def _check_metadata(self, gdf, geometry_column_name='geometry', crs=None):
        assert gdf._geometry_column_name == geometry_column_name
        assert gdf.crs == crs

    def test_merge(self):
        res = self.gdf.merge(self.df, left_on='values', right_on='col1')
        assert isinstance(res, GeoDataFrame)
        assert isinstance(res.geometry, GeoSeries)
        self._check_metadata(res)
        self.gdf.crs = 'epsg:4326'
        self.gdf = self.gdf.rename(columns={'geometry': 'points'}).set_geometry('points')
        res = self.gdf.merge(self.df, left_on='values', right_on='col1')
        assert isinstance(res, GeoDataFrame)
        assert isinstance(res.geometry, GeoSeries)
        self._check_metadata(res, 'points', self.gdf.crs)

    def test_concat_axis0(self):
        res = pd.concat([self.gdf, self.gdf])
        assert res.shape == (6, 2)
        assert isinstance(res, GeoDataFrame)
        assert isinstance(res.geometry, GeoSeries)
        self._check_metadata(res)
        exp = GeoDataFrame(pd.concat([pd.DataFrame(self.gdf), pd.DataFrame(self.gdf)]))
        assert_geodataframe_equal(exp, res)
        res = pd.concat([self.gdf.geometry, self.gdf.geometry])
        assert res.shape == (6,)
        assert isinstance(res, GeoSeries)
        assert isinstance(res.geometry, GeoSeries)

    def test_concat_axis0_crs(self):
        res = pd.concat([self.gdf, self.gdf])
        self._check_metadata(res)
        res1 = pd.concat([self.gdf.set_crs('epsg:4326'), self.gdf.set_crs('epsg:4326')])
        self._check_metadata(res1, crs='epsg:4326')
        with pytest.warns(UserWarning, match='CRS not set for some of the concatenation inputs.*'):
            res2 = pd.concat([self.gdf, self.gdf.set_crs('epsg:4326')])
            self._check_metadata(res2, crs='epsg:4326')
        with pytest.raises(ValueError, match='Cannot determine common CRS for concatenation inputs.*'):
            pd.concat([self.gdf.set_crs('epsg:4326'), self.gdf.set_crs('epsg:4327')])
        with pytest.warns(UserWarning, match='CRS not set for some of the concatenation inputs.*'):
            res3 = pd.concat([self.gdf, self.gdf.set_crs('epsg:4326'), self.gdf.set_crs('epsg:4326')])
            self._check_metadata(res3, crs='epsg:4326')
        with pytest.raises(ValueError, match='Cannot determine common CRS for concatenation inputs.*'):
            pd.concat([self.gdf, self.gdf.set_crs('epsg:4326'), self.gdf.set_crs('epsg:4327')])

    def test_concat_axis0_unaligned_cols(self):
        gdf = self.gdf.set_crs('epsg:4326').assign(geom=self.gdf.geometry.set_crs('epsg:4327'))
        both_geom_cols = gdf[['geom', 'geometry']]
        single_geom_col = gdf[['geometry']]
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            pd.concat([both_geom_cols, single_geom_col])
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            pd.concat([single_geom_col, both_geom_cols])
        explicit_all_none_case = gdf[['geometry']].assign(geom=GeoSeries([None for _ in range(len(gdf))]))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            pd.concat([both_geom_cols, explicit_all_none_case])
        with pytest.warns(UserWarning, match='CRS not set for some of the concatenation inputs.*'):
            partial_none_case = self.gdf[['geometry']]
            partial_none_case.iloc[0] = None
            pd.concat([single_geom_col, partial_none_case])

    def test_concat_axis0_crs_wkt_mismatch(self):
        wkt_template = 'GEOGCRS["WGS 84",\n        ENSEMBLE["World Geodetic System 1984 ensemble",\n        MEMBER["World Geodetic System 1984 (Transit)"],\n        MEMBER["World Geodetic System 1984 (G730)"],\n        MEMBER["World Geodetic System 1984 (G873)"],\n        MEMBER["World Geodetic System 1984 (G1150)"],\n        MEMBER["World Geodetic System 1984 (G1674)"],\n        MEMBER["World Geodetic System 1984 (G1762)"],\n        MEMBER["World Geodetic System 1984 (G2139)"],\n        ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]],\n        ENSEMBLEACCURACY[2.0]],PRIMEM["Greenwich",0,\n        ANGLEUNIT["degree",0.0174532925199433]],CS[ellipsoidal,2],\n        AXIS["geodetic latitude (Lat)",north,ORDER[1],\n        ANGLEUNIT["degree",0.0174532925199433]],\n        AXIS["geodetic longitude (Lon)",east,ORDER[2],\n        ANGLEUNIT["degree",0.0174532925199433]],\n        USAGE[SCOPE["Horizontal component of 3D system."],\n        AREA["World.{}"],BBOX[-90,-180,90,180]],ID["EPSG",4326]]'
        wkt_v1 = wkt_template.format('')
        wkt_v2 = wkt_template.format(' ')
        crs1 = pyproj.CRS.from_wkt(wkt_v1)
        crs2 = pyproj.CRS.from_wkt(wkt_v2)
        assert len({crs1, crs2}) == 2
        assert crs1 == crs2
        expected = pd.concat([self.gdf, self.gdf]).set_crs(crs1)
        res = pd.concat([self.gdf.set_crs(crs1), self.gdf.set_crs(crs2)])
        assert_geodataframe_equal(expected, res)

    def test_concat_axis1(self):
        res = pd.concat([self.gdf, self.df], axis=1)
        assert res.shape == (3, 4)
        assert isinstance(res, GeoDataFrame)
        assert isinstance(res.geometry, GeoSeries)
        self._check_metadata(res)

    def test_concat_axis1_multiple_geodataframes(self):
        if PANDAS_GE_21:
            expected_err = "Concat operation has resulted in multiple columns using the geometry column name 'geometry'."
        else:
            expected_err = "GeoDataFrame does not support multiple columns using the geometry column name 'geometry'"
        with pytest.raises(ValueError, match=expected_err):
            pd.concat([self.gdf, self.gdf], axis=1)
        df2 = self.gdf.rename_geometry('geom')
        expected_err2 = "Concat operation has resulted in multiple columns using the geometry column name 'geom'."
        with pytest.raises(ValueError, match=expected_err2):
            pd.concat([df2, df2], axis=1)
        res3 = pd.concat([df2.set_crs('epsg:4326'), self.gdf], axis=1)
        self._check_metadata(res3, geometry_column_name='geom', crs='epsg:4326')

    @pytest.mark.filterwarnings('ignore:Accessing CRS')
    def test_concat_axis1_geoseries(self):
        gseries2 = GeoSeries([Point(i, i) for i in range(3, 6)], crs='epsg:4326')
        result = pd.concat([gseries2, self.gseries], axis=1)
        assert type(result) is GeoDataFrame
        assert result._geometry_column_name is None
        assert_index_equal(pd.Index([0, 1]), result.columns)
        gseries2.name = 'foo'
        result2 = pd.concat([gseries2, self.gseries], axis=1)
        assert type(result2) is GeoDataFrame
        assert result._geometry_column_name is None
        assert_index_equal(pd.Index(['foo', 0]), result2.columns)