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
def _call_func_on_transformers(self, X, y, func, column_as_labels, routed_params):
    """
        Private function to fit and/or transform on demand.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            The data to be used in fit and/or transform.

        y : array-like of shape (n_samples,)
            Targets.

        func : callable
            Function to call, which can be _fit_transform_one or
            _transform_one.

        column_as_labels : bool
            Used to iterate through transformers. If True, columns are returned
            as strings. If False, columns are returned as they were given by
            the user. Can be True only if the ``ColumnTransformer`` is already
            fitted.

        routed_params : dict
            The routed parameters as the output from ``process_routing``.

        Returns
        -------
        Return value (transformers and/or transformed X data) depends
        on the passed function.
        """
    if func is _fit_transform_one:
        fitted = False
    else:
        fitted = True
    transformers = list(self._iter(fitted=fitted, column_as_labels=column_as_labels, skip_drop=True, skip_empty_columns=True))
    try:
        jobs = []
        for idx, (name, trans, column, weight) in enumerate(transformers, start=1):
            if func is _fit_transform_one:
                if trans == 'passthrough':
                    output_config = _get_output_config('transform', self)
                    trans = FunctionTransformer(accept_sparse=True, check_inverse=False, feature_names_out='one-to-one').set_output(transform=output_config['dense'])
                extra_args = dict(message_clsname='ColumnTransformer', message=self._log_message(name, idx, len(transformers)))
            else:
                extra_args = {}
            jobs.append(delayed(func)(transformer=clone(trans) if not fitted else trans, X=_safe_indexing(X, column, axis=1), y=y, weight=weight, **extra_args, params=routed_params[name]))
        return Parallel(n_jobs=self.n_jobs)(jobs)
    except ValueError as e:
        if 'Expected 2D array, got 1D array instead' in str(e):
            raise ValueError(_ERR_MSG_1DCOLUMN) from e
        else:
            raise