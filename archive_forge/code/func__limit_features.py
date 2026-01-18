import array
import re
import unicodedata
import warnings
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from numbers import Integral
from operator import itemgetter
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin, _fit_context
from ..exceptions import NotFittedError
from ..preprocessing import normalize
from ..utils import _IS_32BIT
from ..utils._param_validation import HasMethods, Interval, RealNotInt, StrOptions
from ..utils.validation import FLOAT_DTYPES, check_array, check_is_fitted
from ._hash import FeatureHasher
from ._stop_words import ENGLISH_STOP_WORDS
def _limit_features(self, X, vocabulary, high=None, low=None, limit=None):
    """Remove too rare or too common features.

        Prune features that are non zero in more samples than high or less
        documents than low, modifying the vocabulary, and restricting it to
        at most the limit most frequent.

        This does not prune samples with zero features.
        """
    if high is None and low is None and (limit is None):
        return (X, set())
    dfs = _document_frequency(X)
    mask = np.ones(len(dfs), dtype=bool)
    if high is not None:
        mask &= dfs <= high
    if low is not None:
        mask &= dfs >= low
    if limit is not None and mask.sum() > limit:
        tfs = np.asarray(X.sum(axis=0)).ravel()
        mask_inds = (-tfs[mask]).argsort()[:limit]
        new_mask = np.zeros(len(dfs), dtype=bool)
        new_mask[np.where(mask)[0][mask_inds]] = True
        mask = new_mask
    new_indices = np.cumsum(mask) - 1
    removed_terms = set()
    for term, old_index in list(vocabulary.items()):
        if mask[old_index]:
            vocabulary[term] = new_indices[old_index]
        else:
            del vocabulary[term]
            removed_terms.add(term)
    kept_indices = np.where(mask)[0]
    if len(kept_indices) == 0:
        raise ValueError('After pruning, no terms remain. Try a lower min_df or a higher max_df.')
    return (X[:, kept_indices], removed_terms)