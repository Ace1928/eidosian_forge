import numbers
import warnings
from collections import Counter
from functools import partial
import numpy as np
import numpy.ma as ma
from scipy import sparse as sp
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import _is_pandas_na, is_scalar_nan
from ..utils._mask import _get_mask
from ..utils._param_validation import MissingValues, StrOptions
from ..utils.fixes import _mode
from ..utils.sparsefuncs import _get_median
from ..utils.validation import FLOAT_DTYPES, _check_feature_names_in, check_is_fitted
def _transform_indicator(self, X):
    """Compute the indicator mask.'

        Note that X must be the original data as passed to the imputer before
        any imputation, since imputation may be done inplace in some cases.
        """
    if self.add_indicator:
        if not hasattr(self, 'indicator_'):
            raise ValueError('Make sure to call _fit_indicator before _transform_indicator')
        return self.indicator_.transform(X)