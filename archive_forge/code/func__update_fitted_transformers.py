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
def _update_fitted_transformers(self, transformers):
    """Set self.transformers_ from given transformers.

        Parameters
        ----------
        transformers : list of estimators
            The fitted estimators as the output of
            `self._call_func_on_transformers(func=_fit_transform_one, ...)`.
            That function doesn't include 'drop' or transformers for which no
            column is selected. 'drop' is kept as is, and for the no-column
            transformers the unfitted transformer is put in
            `self.transformers_`.
        """
    fitted_transformers = iter(transformers)
    transformers_ = []
    for name, old, column, _ in self._iter(fitted=False, column_as_labels=False, skip_drop=False, skip_empty_columns=False):
        if old == 'drop':
            trans = 'drop'
        elif _is_empty_column_selection(column):
            trans = old
        else:
            trans = next(fitted_transformers)
        transformers_.append((name, trans, column))
    assert not list(fitted_transformers)
    self.transformers_ = transformers_