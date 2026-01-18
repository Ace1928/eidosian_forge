import numbers
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from inspect import signature
from itertools import chain, combinations
from math import ceil, floor
import numpy as np
from scipy.special import comb
from ..utils import (
from ..utils._param_validation import Interval, RealNotInt, validate_params
from ..utils.metadata_routing import _MetadataRequester
from ..utils.multiclass import type_of_target
from ..utils.validation import _num_samples, check_array, column_or_1d
class _RepeatedSplits(_MetadataRequester, metaclass=ABCMeta):
    """Repeated splits for an arbitrary randomized CV splitter.

    Repeats splits for cross-validators n times with different randomization
    in each repetition.

    Parameters
    ----------
    cv : callable
        Cross-validator class.

    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.

    random_state : int, RandomState instance or None, default=None
        Passes `random_state` to the arbitrary repeating cross validator.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    **cvargs : additional params
        Constructor parameters for cv. Must not contain random_state
        and shuffle.
    """
    __metadata_request__split = {'groups': metadata_routing.UNUSED}

    def __init__(self, cv, *, n_repeats=10, random_state=None, **cvargs):
        if not isinstance(n_repeats, numbers.Integral):
            raise ValueError('Number of repetitions must be of Integral type.')
        if n_repeats <= 0:
            raise ValueError('Number of repetitions must be greater than 0.')
        if any((key in cvargs for key in ('random_state', 'shuffle'))):
            raise ValueError('cvargs must not contain random_state or shuffle.')
        self.cv = cv
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.cvargs = cvargs

    def split(self, X, y=None, groups=None):
        """Generates indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        n_repeats = self.n_repeats
        rng = check_random_state(self.random_state)
        for idx in range(n_repeats):
            cv = self.cv(random_state=rng, shuffle=True, **self.cvargs)
            for train_index, test_index in cv.split(X, y, groups):
                yield (train_index, test_index)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
            ``np.zeros(n_samples)`` may be used as a placeholder.

        y : object
            Always ignored, exists for compatibility.
            ``np.zeros(n_samples)`` may be used as a placeholder.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        rng = check_random_state(self.random_state)
        cv = self.cv(random_state=rng, shuffle=True, **self.cvargs)
        return cv.get_n_splits(X, y, groups) * self.n_repeats

    def __repr__(self):
        return _build_repr(self)