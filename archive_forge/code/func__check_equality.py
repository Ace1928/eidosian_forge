import warnings
import pandas as pd
from geopandas import GeoDataFrame, GeoSeries
from geopandas.array import GeometryDtype
from geopandas import _vectorized
def _check_equality(left, right, check_less_precise):
    assert_error_message = '{0} out of {1} geometries are not {3}equal.\nIndices where geometries are not {3}equal: {2} \nThe first not {3}equal geometry:\nLeft: {4}\nRight: {5}\n'
    if check_less_precise:
        precise = 'almost '
        equal = _geom_almost_equals_mask(left, right)
    else:
        precise = ''
        equal = _geom_equals_mask(left, right)
    if not equal.all():
        unequal_left_geoms = left[~equal]
        unequal_right_geoms = right[~equal]
        raise AssertionError(assert_error_message.format(len(unequal_left_geoms), len(left), unequal_left_geoms.index.to_list(), precise, _truncated_string(unequal_left_geoms.iloc[0]), _truncated_string(unequal_right_geoms.iloc[0])))