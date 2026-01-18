from array import array
from collections.abc import Iterable, Mapping
from numbers import Number
from operator import itemgetter
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import check_array
from ..utils.validation import check_is_fitted
def _add_iterable_element(self, f, v, feature_names, vocab, *, fitting=True, transforming=False, indices=None, values=None):
    """Add feature names for iterable of strings"""
    for vv in v:
        if isinstance(vv, str):
            feature_name = '%s%s%s' % (f, self.separator, vv)
            vv = 1
        else:
            raise TypeError(f'Unsupported type {type(vv)} in iterable value. Only iterables of string are supported.')
        if fitting and feature_name not in vocab:
            vocab[feature_name] = len(feature_names)
            feature_names.append(feature_name)
        if transforming and feature_name in vocab:
            indices.append(vocab[feature_name])
            values.append(self.dtype(vv))