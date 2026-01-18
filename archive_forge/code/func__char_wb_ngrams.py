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
def _char_wb_ngrams(self, text_document):
    """Whitespace sensitive char-n-gram tokenization.

        Tokenize text_document into a sequence of character n-grams
        operating only inside word boundaries. n-grams at the edges
        of words are padded with space."""
    text_document = self._white_spaces.sub(' ', text_document)
    min_n, max_n = self.ngram_range
    ngrams = []
    ngrams_append = ngrams.append
    for w in text_document.split():
        w = ' ' + w + ' '
        w_len = len(w)
        for n in range(min_n, max_n + 1):
            offset = 0
            ngrams_append(w[offset:offset + n])
            while offset + n < w_len:
                offset += 1
                ngrams_append(w[offset:offset + n])
            if offset == 0:
                break
    return ngrams