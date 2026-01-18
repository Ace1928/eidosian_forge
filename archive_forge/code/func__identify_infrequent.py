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
def _identify_infrequent(self, category_count, n_samples, col_idx):
    """Compute the infrequent indices.

        Parameters
        ----------
        category_count : ndarray of shape (n_cardinality,)
            Category counts.

        n_samples : int
            Number of samples.

        col_idx : int
            Index of the current category. Only used for the error message.

        Returns
        -------
        output : ndarray of shape (n_infrequent_categories,) or None
            If there are infrequent categories, indices of infrequent
            categories. Otherwise None.
        """
    if isinstance(self.min_frequency, numbers.Integral):
        infrequent_mask = category_count < self.min_frequency
    elif isinstance(self.min_frequency, numbers.Real):
        min_frequency_abs = n_samples * self.min_frequency
        infrequent_mask = category_count < min_frequency_abs
    else:
        infrequent_mask = np.zeros(category_count.shape[0], dtype=bool)
    n_current_features = category_count.size - infrequent_mask.sum() + 1
    if self.max_categories is not None and self.max_categories < n_current_features:
        frequent_category_count = self.max_categories - 1
        if frequent_category_count == 0:
            infrequent_mask[:] = True
        else:
            smallest_levels = np.argsort(category_count, kind='mergesort')[:-frequent_category_count]
            infrequent_mask[smallest_levels] = True
    output = np.flatnonzero(infrequent_mask)
    return output if output.size > 0 else None