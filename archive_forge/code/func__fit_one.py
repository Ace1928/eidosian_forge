from collections import defaultdict
from itertools import islice
import numpy as np
from scipy import sparse
from .base import TransformerMixin, _fit_context, clone
from .exceptions import NotFittedError
from .preprocessing import FunctionTransformer
from .utils import Bunch, _print_elapsed_time
from .utils._estimator_html_repr import _VisualBlock
from .utils._metadata_requests import METHODS
from .utils._param_validation import HasMethods, Hidden
from .utils._set_output import (
from .utils._tags import _safe_tags
from .utils.metadata_routing import (
from .utils.metaestimators import _BaseComposition, available_if
from .utils.parallel import Parallel, delayed
from .utils.validation import check_is_fitted, check_memory
def _fit_one(transformer, X, y, weight, message_clsname='', message=None, params=None):
    """
    Fits ``transformer`` to ``X`` and ``y``.
    """
    with _print_elapsed_time(message_clsname, message):
        return transformer.fit(X, y, **params['fit'])