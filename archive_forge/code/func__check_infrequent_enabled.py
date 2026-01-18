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
def _check_infrequent_enabled(self):
    """
        This functions checks whether _infrequent_enabled is True or False.
        This has to be called after parameter validation in the fit function.
        """
    max_categories = getattr(self, 'max_categories', None)
    min_frequency = getattr(self, 'min_frequency', None)
    self._infrequent_enabled = max_categories is not None and max_categories >= 1 or min_frequency is not None