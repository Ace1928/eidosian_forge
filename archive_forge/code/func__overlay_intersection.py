import warnings
from functools import reduce
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import _check_crs, _crs_mismatch_warn
def _overlay_intersection(df1, df2):
    """
    Overlay Intersection operation used in overlay function
    """
    idx1, idx2 = df2.sindex.query(df1.geometry, predicate='intersects', sort=True)
    if idx1.size > 0 and idx2.size > 0:
        left = df1.geometry.take(idx1)
        left.reset_index(drop=True, inplace=True)
        right = df2.geometry.take(idx2)
        right.reset_index(drop=True, inplace=True)
        intersections = left.intersection(right)
        poly_ix = intersections.geom_type.isin(['Polygon', 'MultiPolygon'])
        intersections.loc[poly_ix] = intersections[poly_ix].buffer(0)
        pairs_intersect = pd.DataFrame({'__idx1': idx1, '__idx2': idx2})
        geom_intersect = intersections
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        dfinter = pairs_intersect.merge(df1.drop(df1._geometry_column_name, axis=1), left_on='__idx1', right_index=True)
        dfinter = dfinter.merge(df2.drop(df2._geometry_column_name, axis=1), left_on='__idx2', right_index=True, suffixes=('_1', '_2'))
        return GeoDataFrame(dfinter, geometry=geom_intersect, crs=df1.crs)
    else:
        result = df1.iloc[:0].merge(df2.iloc[:0].drop(df2.geometry.name, axis=1), left_index=True, right_index=True, suffixes=('_1', '_2'))
        result['__idx1'] = np.nan
        result['__idx2'] = np.nan
        return result[result.columns.drop(df1.geometry.name).tolist() + [df1.geometry.name]]