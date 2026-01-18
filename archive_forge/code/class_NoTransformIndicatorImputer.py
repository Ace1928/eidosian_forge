import numpy as np
import pytest
from sklearn.impute._base import _BaseImputer
from sklearn.impute._iterative import _assign_where
from sklearn.utils._mask import _get_mask
from sklearn.utils._testing import _convert_container, assert_allclose
class NoTransformIndicatorImputer(_BaseImputer):

    def fit(self, X, y=None):
        mask = _get_mask(X, value_to_mask=np.nan)
        super()._fit_indicator(mask)
        return self

    def transform(self, X, y=None):
        return self._concatenate_indicator(X, None)