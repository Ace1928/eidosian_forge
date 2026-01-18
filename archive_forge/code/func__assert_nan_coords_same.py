from functools import partial
import numpy as np
import shapely
def _assert_nan_coords_same(x, y, tolerance, err_msg, verbose):
    x, y = np.broadcast_arrays(x, y)
    x_coords = shapely.get_coordinates(x, include_z=True)
    y_coords = shapely.get_coordinates(y, include_z=True)
    if x_coords.shape != y_coords.shape:
        return False
    x_id = np.isnan(x_coords)
    y_id = np.isnan(y_coords)
    if not (x_id == y_id).all():
        msg = build_err_msg([x, y], err_msg + '\nx and y nan coordinate location mismatch:', verbose=verbose)
        raise AssertionError(msg)
    x_no_nan = shapely.transform(x, _replace_nan, include_z=True)
    y_no_nan = shapely.transform(y, _replace_nan, include_z=True)
    return _equals_exact_with_ndim(x_no_nan, y_no_nan, tolerance=tolerance)