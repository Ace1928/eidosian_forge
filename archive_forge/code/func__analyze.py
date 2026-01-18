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
def _analyze(doc, analyzer=None, tokenizer=None, ngrams=None, preprocessor=None, decoder=None, stop_words=None):
    """Chain together an optional series of text processing steps to go from
    a single document to ngrams, with or without tokenizing or preprocessing.

    If analyzer is used, only the decoder argument is used, as the analyzer is
    intended to replace the preprocessor, tokenizer, and ngrams steps.

    Parameters
    ----------
    analyzer: callable, default=None
    tokenizer: callable, default=None
    ngrams: callable, default=None
    preprocessor: callable, default=None
    decoder: callable, default=None
    stop_words: list, default=None

    Returns
    -------
    ngrams: list
        A sequence of tokens, possibly with pairs, triples, etc.
    """
    if decoder is not None:
        doc = decoder(doc)
    if analyzer is not None:
        doc = analyzer(doc)
    else:
        if preprocessor is not None:
            doc = preprocessor(doc)
        if tokenizer is not None:
            doc = tokenizer(doc)
        if ngrams is not None:
            if stop_words is not None:
                doc = ngrams(doc, stop_words)
            else:
                doc = ngrams(doc)
    return doc