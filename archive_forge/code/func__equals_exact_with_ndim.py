from functools import partial
import numpy as np
import shapely
def _equals_exact_with_ndim(x, y, tolerance):
    dimension_equals = shapely.get_coordinate_dimension(x) == shapely.get_coordinate_dimension(y)
    with np.errstate(invalid='ignore'):
        geometry_equals = shapely.equals_exact(x, y, tolerance=tolerance)
    return dimension_equals & geometry_equals