import warnings
import numpy as np
from ..base import BaseEstimator, RegressorMixin, _fit_context, clone
from ..exceptions import NotFittedError
from ..preprocessing import FunctionTransformer
from ..utils import _safe_indexing, check_array
from ..utils._param_validation import HasMethods
from ..utils._tags import _safe_tags
from ..utils.metadata_routing import (
from ..utils.validation import check_is_fitted
def _fit_transformer(self, y):
    """Check transformer and fit transformer.

        Create the default transformer, fit it and make additional inverse
        check on a subset (optional).

        """
    if self.transformer is not None and (self.func is not None or self.inverse_func is not None):
        raise ValueError("'transformer' and functions 'func'/'inverse_func' cannot both be set.")
    elif self.transformer is not None:
        self.transformer_ = clone(self.transformer)
    else:
        if self.func is not None and self.inverse_func is None:
            raise ValueError("When 'func' is provided, 'inverse_func' must also be provided")
        self.transformer_ = FunctionTransformer(func=self.func, inverse_func=self.inverse_func, validate=True, check_inverse=self.check_inverse)
    self.transformer_.fit(y)
    if self.check_inverse:
        idx_selected = slice(None, None, max(1, y.shape[0] // 10))
        y_sel = _safe_indexing(y, idx_selected)
        y_sel_t = self.transformer_.transform(y_sel)
        if not np.allclose(y_sel, self.transformer_.inverse_transform(y_sel_t)):
            warnings.warn("The provided functions or transformer are not strictly inverse of each other. If you are sure you want to proceed regardless, set 'check_inverse=False'", UserWarning)