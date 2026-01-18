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
def _check_get_feature_name_combiner(self):
    if self.feature_name_combiner == 'concat':
        return lambda feature, category: feature + '_' + str(category)
    else:
        dry_run_combiner = self.feature_name_combiner('feature', 'category')
        if not isinstance(dry_run_combiner, str):
            raise TypeError(f'When `feature_name_combiner` is a callable, it should return a Python string. Got {type(dry_run_combiner)} instead.')
        return self.feature_name_combiner