import numpy as np
from scipy.optimize import linear_sum_assignment
from ...utils._param_validation import StrOptions, validate_params
from ...utils.validation import check_array, check_consistent_length
def _check_rows_and_columns(a, b):
    """Unpacks the row and column arrays and checks their shape."""
    check_consistent_length(*a)
    check_consistent_length(*b)
    checks = lambda x: check_array(x, ensure_2d=False)
    a_rows, a_cols = map(checks, a)
    b_rows, b_cols = map(checks, b)
    return (a_rows, a_cols, b_rows, b_cols)