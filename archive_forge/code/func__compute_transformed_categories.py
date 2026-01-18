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
def _compute_transformed_categories(self, i, remove_dropped=True):
    """Compute the transformed categories used for column `i`.

        1. If there are infrequent categories, the category is named
        'infrequent_sklearn'.
        2. Dropped columns are removed when remove_dropped=True.
        """
    cats = self.categories_[i]
    if self._infrequent_enabled:
        infreq_map = self._default_to_infrequent_mappings[i]
        if infreq_map is not None:
            frequent_mask = infreq_map < infreq_map.max()
            infrequent_cat = 'infrequent_sklearn'
            cats = np.concatenate((cats[frequent_mask], np.array([infrequent_cat], dtype=object)))
    if remove_dropped:
        cats = self._remove_dropped_categories(cats, i)
    return cats