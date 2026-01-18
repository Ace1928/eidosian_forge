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
class LeaveOneGroupOut(GroupsConsumerMixin, BaseCrossValidator):
    """Leave One Group Out cross-validator.

    Provides train/test indices to split data such that each training set is
    comprised of all samples except ones belonging to one specific group.
    Arbitrary domain specific group information is provided an array integers
    that encodes the group of each sample.

    For instance the groups could be the year of collection of the samples
    and thus allow for cross-validation against time-based splits.

    Read more in the :ref:`User Guide <leave_one_group_out>`.

    Notes
    -----
    Splits are ordered according to the index of the group left out. The first
    split has testing set consisting of the group whose index in `groups` is
    lowest, and so on.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import LeaveOneGroupOut
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([1, 2, 1, 2])
    >>> groups = np.array([1, 1, 2, 2])
    >>> logo = LeaveOneGroupOut()
    >>> logo.get_n_splits(X, y, groups)
    2
    >>> logo.get_n_splits(groups=groups)  # 'groups' is always required
    2
    >>> print(logo)
    LeaveOneGroupOut()
    >>> for i, (train_index, test_index) in enumerate(logo.split(X, y, groups)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}, group={groups[train_index]}")
    ...     print(f"  Test:  index={test_index}, group={groups[test_index]}")
    Fold 0:
      Train: index=[2 3], group=[2 2]
      Test:  index=[0 1], group=[1 1]
    Fold 1:
      Train: index=[0 1], group=[1 1]
      Test:  index=[2 3], group=[2 2]

    See also
    --------
    GroupKFold: K-fold iterator variant with non-overlapping groups.
    """

    def _iter_test_masks(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, input_name='groups', copy=True, ensure_2d=False, dtype=None)
        unique_groups = np.unique(groups)
        if len(unique_groups) <= 1:
            raise ValueError('The groups parameter contains fewer than 2 unique groups (%s). LeaveOneGroupOut expects at least 2.' % unique_groups)
        for i in unique_groups:
            yield (groups == i)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set. This 'groups' parameter must always be specified to
            calculate the number of splits, though the other parameters can be
            omitted.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, input_name='groups', ensure_2d=False, dtype=None)
        return len(np.unique(groups))

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        return super().split(X, y, groups)