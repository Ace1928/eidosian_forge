from functools import partial
import numpy as np
import shapely
def _assert_none_same(x, y, err_msg, verbose):
    x_id = shapely.is_missing(x)
    y_id = shapely.is_missing(y)
    if not (x_id == y_id).all():
        msg = build_err_msg([x, y], err_msg + '\nx and y None location mismatch:', verbose=verbose)
        raise AssertionError(msg)
    if x.ndim == 0:
        return bool(x_id)
    elif y.ndim == 0:
        return bool(y_id)
    else:
        return y_id