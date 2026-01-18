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
class GenericUnivariateSelect(_BaseFilter):
    """Univariate feature selector with configurable strategy.

    Read more in the :ref:`User Guide <univariate_feature_selection>`.

    Parameters
    ----------
    score_func : callable, default=f_classif
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues). For modes 'percentile' or 'kbest' it can return
        a single array scores.

    mode : {'percentile', 'k_best', 'fpr', 'fdr', 'fwe'}, default='percentile'
        Feature selection mode. Note that the `'percentile'` and `'kbest'`
        modes are supporting unsupervised feature selection (when `y` is `None`).

    param : "all", float or int, default=1e-5
        Parameter of the corresponding mode.

    Attributes
    ----------
    scores_ : array-like of shape (n_features,)
        Scores of features.

    pvalues_ : array-like of shape (n_features,)
        p-values of feature scores, None if `score_func` returned scores only.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    f_classif : ANOVA F-value between label/feature for classification tasks.
    mutual_info_classif : Mutual information for a discrete target.
    chi2 : Chi-squared stats of non-negative features for classification tasks.
    f_regression : F-value between label/feature for regression tasks.
    mutual_info_regression : Mutual information for a continuous target.
    SelectPercentile : Select features based on percentile of the highest
        scores.
    SelectKBest : Select features based on the k highest scores.
    SelectFpr : Select features based on a false positive rate test.
    SelectFdr : Select features based on an estimated false discovery rate.
    SelectFwe : Select features based on family-wise error rate.

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.feature_selection import GenericUnivariateSelect, chi2
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> X.shape
    (569, 30)
    >>> transformer = GenericUnivariateSelect(chi2, mode='k_best', param=20)
    >>> X_new = transformer.fit_transform(X, y)
    >>> X_new.shape
    (569, 20)
    """
    _selection_modes: dict = {'percentile': SelectPercentile, 'k_best': SelectKBest, 'fpr': SelectFpr, 'fdr': SelectFdr, 'fwe': SelectFwe}
    _parameter_constraints: dict = {**_BaseFilter._parameter_constraints, 'mode': [StrOptions(set(_selection_modes.keys()))], 'param': [Interval(Real, 0, None, closed='left'), StrOptions({'all'})]}

    def __init__(self, score_func=f_classif, *, mode='percentile', param=1e-05):
        super().__init__(score_func=score_func)
        self.mode = mode
        self.param = param

    def _make_selector(self):
        selector = self._selection_modes[self.mode](score_func=self.score_func)
        possible_params = selector._get_param_names()
        possible_params.remove('score_func')
        selector.set_params(**{possible_params[0]: self.param})
        return selector

    def _more_tags(self):
        return {'preserves_dtype': [np.float64, np.float32]}

    def _check_params(self, X, y):
        self._make_selector()._check_params(X, y)

    def _get_support_mask(self):
        check_is_fitted(self)
        selector = self._make_selector()
        selector.pvalues_ = self.pvalues_
        selector.scores_ = self.scores_
        return selector._get_support_mask()