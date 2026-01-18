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
def _hstack(self, Xs, *, n_samples):
    """Stacks Xs horizontally.

        This allows subclasses to control the stacking behavior, while reusing
        everything else from ColumnTransformer.

        Parameters
        ----------
        Xs : list of {array-like, sparse matrix, dataframe}
            The container to concatenate.
        n_samples : int
            The number of samples in the input data to checking the transformation
            consistency.
        """
    if self.sparse_output_:
        try:
            converted_Xs = [check_array(X, accept_sparse=True, force_all_finite=False) for X in Xs]
        except ValueError as e:
            raise ValueError('For a sparse output, all columns should be a numeric or convertible to a numeric.') from e
        return sparse.hstack(converted_Xs).tocsr()
    else:
        Xs = [f.toarray() if sparse.issparse(f) else f for f in Xs]
        adapter = _get_container_adapter('transform', self)
        if adapter and all((adapter.is_supported_container(X) for X in Xs)):
            transformer_names = [t[0] for t in self._iter(fitted=True, column_as_labels=False, skip_drop=True, skip_empty_columns=True)]
            feature_names_outs = [X.columns for X in Xs if X.shape[1] != 0]
            if self.verbose_feature_names_out:
                feature_names_outs = self._add_prefix_for_feature_names_out(list(zip(transformer_names, feature_names_outs)))
            else:
                feature_names_outs = list(chain.from_iterable(feature_names_outs))
                feature_names_count = Counter(feature_names_outs)
                if any((count > 1 for count in feature_names_count.values())):
                    duplicated_feature_names = sorted((name for name, count in feature_names_count.items() if count > 1))
                    err_msg = f'Duplicated feature names found before concatenating the outputs of the transformers: {duplicated_feature_names}.\n'
                    for transformer_name, X in zip(transformer_names, Xs):
                        if X.shape[1] == 0:
                            continue
                        dup_cols_in_transformer = sorted(set(X.columns).intersection(duplicated_feature_names))
                        if len(dup_cols_in_transformer):
                            err_msg += f'Transformer {transformer_name} has conflicting columns names: {dup_cols_in_transformer}.\n'
                    raise ValueError(err_msg + 'Either make sure that the transformers named above do not generate columns with conflicting names or set verbose_feature_names_out=True to automatically prefix to the output feature names with the name of the transformer to prevent any conflicting names.')
            names_idx = 0
            for X in Xs:
                if X.shape[1] == 0:
                    continue
                names_out = feature_names_outs[names_idx:names_idx + X.shape[1]]
                adapter.rename_columns(X, names_out)
                names_idx += X.shape[1]
            output = adapter.hstack(Xs)
            output_samples = output.shape[0]
            if output_samples != n_samples:
                raise ValueError("Concatenating DataFrames from the transformer's output lead to an inconsistent number of samples. The output may have Pandas Indexes that do not match, or that transformers are returning number of samples which are not the same as the number input samples.")
            return output
        return np.hstack(Xs)