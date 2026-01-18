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
def _get_feature_name_out_for_transformer(self, name, trans, feature_names_in):
    """Gets feature names of transformer.

        Used in conjunction with self._iter(fitted=True) in get_feature_names_out.
        """
    column_indices = self._transformer_to_input_indices[name]
    names = feature_names_in[column_indices]
    if not hasattr(trans, 'get_feature_names_out'):
        raise AttributeError(f'Transformer {name} (type {type(trans).__name__}) does not provide get_feature_names_out.')
    return trans.get_feature_names_out(names)