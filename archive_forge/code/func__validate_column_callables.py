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
def _validate_column_callables(self, X):
    """
        Converts callable column specifications.

        This stores a dictionary of the form `{step_name: column_indices}` and
        calls the `columns` on `X` if `columns` is a callable for a given
        transformer.

        The results are then stored in `self._transformer_to_input_indices`.
        """
    all_columns = []
    transformer_to_input_indices = {}
    for name, _, columns in self.transformers:
        if callable(columns):
            columns = columns(X)
        all_columns.append(columns)
        transformer_to_input_indices[name] = _get_column_indices(X, columns)
    self._columns = all_columns
    self._transformer_to_input_indices = transformer_to_input_indices