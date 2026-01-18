from itertools import chain
from numbers import Integral
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils._param_validation import Interval, StrOptions
from ._hashing_fast import transform as _hashing_transform
Transform a sequence of instances to a scipy.sparse matrix.

        Parameters
        ----------
        raw_X : iterable over iterable over raw features, length = n_samples
            Samples. Each sample must be iterable an (e.g., a list or tuple)
            containing/generating feature names (and optionally values, see
            the input_type constructor argument) which will be hashed.
            raw_X need not support the len function, so it can be the result
            of a generator; n_samples is determined on the fly.

        Returns
        -------
        X : sparse matrix of shape (n_samples, n_features)
            Feature matrix, for use with estimators or further transformers.
        