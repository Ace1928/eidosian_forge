import numpy as np
from scipy import linalg
from ..utils import check_array
from ..utils._param_validation import StrOptions
from ..utils.extmath import row_norms
from ._base import BaseMixture, _check_shape
def _check_precisions_full(precisions, covariance_type):
    """Check the precision matrices are symmetric and positive-definite."""
    for prec in precisions:
        _check_precision_matrix(prec, covariance_type)