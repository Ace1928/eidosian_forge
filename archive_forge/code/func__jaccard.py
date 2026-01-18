import numpy as np
from scipy.optimize import linear_sum_assignment
from ...utils._param_validation import StrOptions, validate_params
from ...utils.validation import check_array, check_consistent_length
def _jaccard(a_rows, a_cols, b_rows, b_cols):
    """Jaccard coefficient on the elements of the two biclusters."""
    intersection = (a_rows * b_rows).sum() * (a_cols * b_cols).sum()
    a_size = a_rows.sum() * a_cols.sum()
    b_size = b_rows.sum() * b_cols.sum()
    return intersection / (a_size + b_size - intersection)