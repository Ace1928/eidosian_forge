from snappy.SnapPy import matrix, vector
from snappy.hyperboloid import (r13_dot,
def _orthonormalize_row_sane(row, fallback_value, other_rows, sign):
    r = _orthonormalize_row(row, other_rows, sign)
    if r is not None:
        return r
    r = _orthonormalize_row(fallback_value, other_rows, sign)
    if r is not None:
        return r
    return fallback_value