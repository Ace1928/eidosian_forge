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
class GroupShuffleSplit(GroupsConsumerMixin, ShuffleSplit):
    """Shuffle-Group(s)-Out cross-validation iterator.

    Provides randomized train/test indices to split data according to a
    third-party provided group. This group information can be used to encode
    arbitrary domain specific stratifications of the samples as integers.

    For instance the groups could be the year of collection of the samples
    and thus allow for cross-validation against time-based splits.

    The difference between LeavePGroupsOut and GroupShuffleSplit is that
    the former generates splits using all subsets of size ``p`` unique groups,
    whereas GroupShuffleSplit generates a user-determined number of random
    test splits, each with a user-determined fraction of unique groups.

    For example, a less computationally intensive alternative to
    ``LeavePGroupsOut(p=10)`` would be
    ``GroupShuffleSplit(test_size=10, n_splits=100)``.

    Note: The parameters ``test_size`` and ``train_size`` refer to groups, and
    not to samples, as in ShuffleSplit.

    Read more in the :ref:`User Guide <group_shuffle_split>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    Parameters
    ----------
    n_splits : int, default=5
        Number of re-shuffling & splitting iterations.

    test_size : float, int, default=0.2
        If float, should be between 0.0 and 1.0 and represent the proportion
        of groups to include in the test split (rounded up). If int,
        represents the absolute number of test groups. If None, the value is
        set to the complement of the train size.
        The default will change in version 0.21. It will remain 0.2 only
        if ``train_size`` is unspecified, otherwise it will complement
        the specified ``train_size``.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the groups to include in the train split. If
        int, represents the absolute number of train groups. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import GroupShuffleSplit
    >>> X = np.ones(shape=(8, 2))
    >>> y = np.ones(shape=(8, 1))
    >>> groups = np.array([1, 1, 2, 2, 2, 3, 3, 3])
    >>> print(groups.shape)
    (8,)
    >>> gss = GroupShuffleSplit(n_splits=2, train_size=.7, random_state=42)
    >>> gss.get_n_splits()
    2
    >>> print(gss)
    GroupShuffleSplit(n_splits=2, random_state=42, test_size=None, train_size=0.7)
    >>> for i, (train_index, test_index) in enumerate(gss.split(X, y, groups)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}, group={groups[train_index]}")
    ...     print(f"  Test:  index={test_index}, group={groups[test_index]}")
    Fold 0:
      Train: index=[2 3 4 5 6 7], group=[2 2 2 3 3 3]
      Test:  index=[0 1], group=[1 1]
    Fold 1:
      Train: index=[0 1 5 6 7], group=[1 1 3 3 3]
      Test:  index=[2 3 4], group=[2 2 2]

    See Also
    --------
    ShuffleSplit : Shuffles samples to create independent test/train sets.

    LeavePGroupsOut : Train set leaves out all possible subsets of `p` groups.
    """

    def __init__(self, n_splits=5, *, test_size=None, train_size=None, random_state=None):
        super().__init__(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=random_state)
        self._default_test_size = 0.2

    def _iter_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, input_name='groups', ensure_2d=False, dtype=None)
        classes, group_indices = np.unique(groups, return_inverse=True)
        for group_train, group_test in super()._iter_indices(X=classes):
            train = np.flatnonzero(np.isin(group_indices, group_train))
            test = np.flatnonzero(np.isin(group_indices, group_test))
            yield (train, test)

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

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        return super().split(X, y, groups)