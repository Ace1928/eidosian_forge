from array import array
from collections.abc import Iterable, Mapping
from numbers import Number
from operator import itemgetter
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import check_array
from ..utils.validation import check_is_fitted
Restrict the features to those in support using feature selection.

        This function modifies the estimator in-place.

        Parameters
        ----------
        support : array-like
            Boolean mask or list of indices (as returned by the get_support
            member of feature selectors).
        indices : bool, default=False
            Whether support is a list of indices.

        Returns
        -------
        self : object
            DictVectorizer class instance.

        Examples
        --------
        >>> from sklearn.feature_extraction import DictVectorizer
        >>> from sklearn.feature_selection import SelectKBest, chi2
        >>> v = DictVectorizer()
        >>> D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
        >>> X = v.fit_transform(D)
        >>> support = SelectKBest(chi2, k=2).fit(X, [0, 1])
        >>> v.get_feature_names_out()
        array(['bar', 'baz', 'foo'], ...)
        >>> v.restrict(support.get_support())
        DictVectorizer()
        >>> v.get_feature_names_out()
        array(['bar', 'foo'], ...)
        