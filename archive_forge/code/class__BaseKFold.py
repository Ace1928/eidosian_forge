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
class _BaseKFold(BaseCrossValidator, metaclass=ABCMeta):
    """Base class for K-Fold cross-validators and TimeSeriesSplit."""

    @abstractmethod
    def __init__(self, n_splits, *, shuffle, random_state):
        if not isinstance(n_splits, numbers.Integral):
            raise ValueError('The number of folds must be of Integral type. %s of type %s was passed.' % (n_splits, type(n_splits)))
        n_splits = int(n_splits)
        if n_splits <= 1:
            raise ValueError('k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits={0}.'.format(n_splits))
        if not isinstance(shuffle, bool):
            raise TypeError('shuffle must be True or False; got {0}'.format(shuffle))
        if not shuffle and random_state is not None:
            raise ValueError('Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.')
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
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
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        if self.n_splits > n_samples:
            raise ValueError('Cannot have number of splits n_splits={0} greater than the number of samples: n_samples={1}.'.format(self.n_splits, n_samples))
        for train, test in super().split(X, y, groups):
            yield (train, test)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits