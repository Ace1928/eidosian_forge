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
def _count_vocab(self, raw_documents, fixed_vocab):
    """Create sparse feature matrix, and vocabulary where fixed_vocab=False"""
    if fixed_vocab:
        vocabulary = self.vocabulary_
    else:
        vocabulary = defaultdict()
        vocabulary.default_factory = vocabulary.__len__
    analyze = self.build_analyzer()
    j_indices = []
    indptr = []
    values = _make_int_array()
    indptr.append(0)
    for doc in raw_documents:
        feature_counter = {}
        for feature in analyze(doc):
            try:
                feature_idx = vocabulary[feature]
                if feature_idx not in feature_counter:
                    feature_counter[feature_idx] = 1
                else:
                    feature_counter[feature_idx] += 1
            except KeyError:
                continue
        j_indices.extend(feature_counter.keys())
        values.extend(feature_counter.values())
        indptr.append(len(j_indices))
    if not fixed_vocab:
        vocabulary = dict(vocabulary)
        if not vocabulary:
            raise ValueError('empty vocabulary; perhaps the documents only contain stop words')
    if indptr[-1] > np.iinfo(np.int32).max:
        if _IS_32BIT:
            raise ValueError('sparse CSR array has {} non-zero elements and requires 64 bit indexing, which is unsupported with 32 bit Python.'.format(indptr[-1]))
        indices_dtype = np.int64
    else:
        indices_dtype = np.int32
    j_indices = np.asarray(j_indices, dtype=indices_dtype)
    indptr = np.asarray(indptr, dtype=indices_dtype)
    values = np.frombuffer(values, dtype=np.intc)
    X = sp.csr_matrix((values, j_indices, indptr), shape=(len(indptr) - 1, len(vocabulary)), dtype=self.dtype)
    X.sort_indices()
    return (vocabulary, X)