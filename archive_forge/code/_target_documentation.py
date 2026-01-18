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
Number of features seen during :term:`fit`.