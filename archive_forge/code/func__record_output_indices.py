import warnings
from collections import Counter
from itertools import chain
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from ..base import TransformerMixin, _fit_context, clone
from ..pipeline import _fit_transform_one, _name_estimators, _transform_one
from ..preprocessing import FunctionTransformer
from ..utils import Bunch, _get_column_indices, _safe_indexing
from ..utils._estimator_html_repr import _VisualBlock
from ..utils._metadata_requests import METHODS
from ..utils._param_validation import HasMethods, Hidden, Interval, StrOptions
from ..utils._set_output import (
from ..utils.metadata_routing import (
from ..utils.metaestimators import _BaseComposition
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
def _record_output_indices(self, Xs):
    """
        Record which transformer produced which column.
        """
    idx = 0
    self.output_indices_ = {}
    for transformer_idx, (name, _, _, _) in enumerate(self._iter(fitted=True, column_as_labels=False, skip_drop=True, skip_empty_columns=True)):
        n_columns = Xs[transformer_idx].shape[1]
        self.output_indices_[name] = slice(idx, idx + n_columns)
        idx += n_columns
    all_names = [t[0] for t in self.transformers] + ['remainder']
    for name in all_names:
        if name not in self.output_indices_:
            self.output_indices_[name] = slice(0, 0)