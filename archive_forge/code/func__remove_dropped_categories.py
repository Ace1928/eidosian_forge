import numbers
import warnings
from numbers import Integral
import numpy as np
from scipy import sparse
from ..base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin, _fit_context
from ..utils import _safe_indexing, check_array, is_scalar_nan
from ..utils._encode import _check_unknown, _encode, _get_counts, _unique
from ..utils._mask import _get_mask
from ..utils._param_validation import Interval, RealNotInt, StrOptions
from ..utils._set_output import _get_output_config
from ..utils.validation import _check_feature_names_in, check_is_fitted
def _remove_dropped_categories(self, categories, i):
    """Remove dropped categories."""
    if self._drop_idx_after_grouping is not None and self._drop_idx_after_grouping[i] is not None:
        return np.delete(categories, self._drop_idx_after_grouping[i])
    return categories