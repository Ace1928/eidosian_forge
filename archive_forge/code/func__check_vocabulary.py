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
def _check_vocabulary(self):
    """Check if vocabulary is empty or missing (not fitted)"""
    if not hasattr(self, 'vocabulary_'):
        self._validate_vocabulary()
        if not self.fixed_vocabulary_:
            raise NotFittedError('Vocabulary not fitted or provided')
    if len(self.vocabulary_) == 0:
        raise ValueError('Vocabulary is empty')