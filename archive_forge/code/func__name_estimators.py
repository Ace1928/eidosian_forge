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
def _name_estimators(estimators):
    """Generate names for estimators."""
    names = [estimator if isinstance(estimator, str) else type(estimator).__name__.lower() for estimator in estimators]
    namecount = defaultdict(int)
    for est, name in zip(estimators, names):
        namecount[name] += 1
    for k, v in list(namecount.items()):
        if v == 1:
            del namecount[k]
    for i in reversed(range(len(estimators))):
        name = names[i]
        if name in namecount:
            names[i] += '-%d' % namecount[name]
            namecount[name] -= 1
    return list(zip(names, estimators))