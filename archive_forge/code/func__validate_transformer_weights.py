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
def _validate_transformer_weights(self):
    if not self.transformer_weights:
        return
    transformer_names = set((name for name, _ in self.transformer_list))
    for name in self.transformer_weights:
        if name not in transformer_names:
            raise ValueError(f'Attempting to weight transformer "{name}", but it is not present in transformer_list.')