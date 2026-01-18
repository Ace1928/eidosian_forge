import warnings
from numbers import Integral, Real
import numpy as np
from scipy import special, stats
from scipy.sparse import issparse
from ..base import BaseEstimator, _fit_context
from ..preprocessing import LabelBinarizer
from ..utils import as_float_array, check_array, check_X_y, safe_mask, safe_sqr
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.validation import check_is_fitted
from ._base import SelectorMixin
class _BaseFilter(SelectorMixin, BaseEstimator):
    """Initialize the univariate feature selection.

    Parameters
    ----------
    score_func : callable
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues) or a single array with scores.
    """
    _parameter_constraints: dict = {'score_func': [callable]}

    def __init__(self, score_func):
        self.score_func = score_func

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Run score function on (X, y) and get the appropriate features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,) or None
            The target values (class labels in classification, real numbers in
            regression). If the selector is unsupervised then `y` can be set to `None`.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if y is None:
            X = self._validate_data(X, accept_sparse=['csr', 'csc'])
        else:
            X, y = self._validate_data(X, y, accept_sparse=['csr', 'csc'], multi_output=True)
        self._check_params(X, y)
        score_func_ret = self.score_func(X, y)
        if isinstance(score_func_ret, (list, tuple)):
            self.scores_, self.pvalues_ = score_func_ret
            self.pvalues_ = np.asarray(self.pvalues_)
        else:
            self.scores_ = score_func_ret
            self.pvalues_ = None
        self.scores_ = np.asarray(self.scores_)
        return self

    def _check_params(self, X, y):
        pass

    def _more_tags(self):
        return {'requires_y': True}