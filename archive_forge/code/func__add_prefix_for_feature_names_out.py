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
def _add_prefix_for_feature_names_out(self, transformer_with_feature_names_out):
    """Add prefix for feature names out that includes the transformer names.

        Parameters
        ----------
        transformer_with_feature_names_out : list of tuples of (str, array-like of str)
            The tuple consistent of the transformer's name and its feature names out.

        Returns
        -------
        feature_names_out : ndarray of shape (n_features,), dtype=str
            Transformed feature names.
        """
    if self.verbose_feature_names_out:
        names = list(chain.from_iterable(((f'{name}__{i}' for i in feature_names_out) for name, feature_names_out in transformer_with_feature_names_out)))
        return np.asarray(names, dtype=object)
    feature_names_count = Counter(chain.from_iterable((s for _, s in transformer_with_feature_names_out)))
    top_6_overlap = [name for name, count in feature_names_count.most_common(6) if count > 1]
    top_6_overlap.sort()
    if top_6_overlap:
        if len(top_6_overlap) == 6:
            names_repr = str(top_6_overlap[:5])[:-1] + ', ...]'
        else:
            names_repr = str(top_6_overlap)
        raise ValueError(f'Output feature names: {names_repr} are not unique. Please set verbose_feature_names_out=True to add prefixes to feature names')
    return np.concatenate([name for _, name in transformer_with_feature_names_out])