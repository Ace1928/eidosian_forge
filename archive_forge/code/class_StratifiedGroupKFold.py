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
class StratifiedGroupKFold(GroupsConsumerMixin, _BaseKFold):
    """Stratified K-Fold iterator variant with non-overlapping groups.

    This cross-validation object is a variation of StratifiedKFold attempts to
    return stratified folds with non-overlapping groups. The folds are made by
    preserving the percentage of samples for each class.

    Each group will appear exactly once in the test set across all folds (the
    number of distinct groups has to be at least equal to the number of folds).

    The difference between :class:`~sklearn.model_selection.GroupKFold`
    and :class:`~sklearn.model_selection.StratifiedGroupKFold` is that
    the former attempts to create balanced folds such that the number of
    distinct groups is approximately the same in each fold, whereas
    StratifiedGroupKFold attempts to create folds which preserve the
    percentage of samples for each class as much as possible given the
    constraint of non-overlapping groups between splits.

    Read more in the :ref:`User Guide <cross_validation>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
        This implementation can only shuffle groups that have approximately the
        same y distribution, no global shuffle will be performed.

    random_state : int or RandomState instance, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold for each class.
        Otherwise, leave `random_state` as `None`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedGroupKFold
    >>> X = np.ones((17, 2))
    >>> y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> groups = np.array([1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7, 8, 8])
    >>> sgkf = StratifiedGroupKFold(n_splits=3)
    >>> sgkf.get_n_splits(X, y)
    3
    >>> print(sgkf)
    StratifiedGroupKFold(n_splits=3, random_state=None, shuffle=False)
    >>> for i, (train_index, test_index) in enumerate(sgkf.split(X, y, groups)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"         group={groups[train_index]}")
    ...     print(f"  Test:  index={test_index}")
    ...     print(f"         group={groups[test_index]}")
    Fold 0:
      Train: index=[ 0  1  2  3  7  8  9 10 11 15 16]
             group=[1 1 2 2 4 5 5 5 5 8 8]
      Test:  index=[ 4  5  6 12 13 14]
             group=[3 3 3 6 6 7]
    Fold 1:
      Train: index=[ 4  5  6  7  8  9 10 11 12 13 14]
             group=[3 3 3 4 5 5 5 5 6 6 7]
      Test:  index=[ 0  1  2  3 15 16]
             group=[1 1 2 2 8 8]
    Fold 2:
      Train: index=[ 0  1  2  3  4  5  6 12 13 14 15 16]
             group=[1 1 2 2 3 3 3 6 6 7 8 8]
      Test:  index=[ 7  8  9 10 11]
             group=[4 5 5 5 5]

    Notes
    -----
    The implementation is designed to:

    * Mimic the behavior of StratifiedKFold as much as possible for trivial
      groups (e.g. when each group contains only one sample).
    * Be invariant to class label: relabelling ``y = ["Happy", "Sad"]`` to
      ``y = [1, 0]`` should not change the indices generated.
    * Stratify based on samples as much as possible while keeping
      non-overlapping groups constraint. That means that in some cases when
      there is a small number of groups containing a large number of samples
      the stratification will not be possible and the behavior will be close
      to GroupKFold.

    See also
    --------
    StratifiedKFold: Takes class information into account to build folds which
        retain class distributions (for binary or multiclass classification
        tasks).

    GroupKFold: K-fold iterator variant with non-overlapping groups.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(self, X, y, groups):
        rng = check_random_state(self.random_state)
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        allowed_target_types = ('binary', 'multiclass')
        if type_of_target_y not in allowed_target_types:
            raise ValueError('Supported target types are: {}. Got {!r} instead.'.format(allowed_target_types, type_of_target_y))
        y = column_or_1d(y)
        _, y_inv, y_cnt = np.unique(y, return_inverse=True, return_counts=True)
        if np.all(self.n_splits > y_cnt):
            raise ValueError('n_splits=%d cannot be greater than the number of members in each class.' % self.n_splits)
        n_smallest_class = np.min(y_cnt)
        if self.n_splits > n_smallest_class:
            warnings.warn('The least populated class in y has only %d members, which is less than n_splits=%d.' % (n_smallest_class, self.n_splits), UserWarning)
        n_classes = len(y_cnt)
        _, groups_inv, groups_cnt = np.unique(groups, return_inverse=True, return_counts=True)
        y_counts_per_group = np.zeros((len(groups_cnt), n_classes))
        for class_idx, group_idx in zip(y_inv, groups_inv):
            y_counts_per_group[group_idx, class_idx] += 1
        y_counts_per_fold = np.zeros((self.n_splits, n_classes))
        groups_per_fold = defaultdict(set)
        if self.shuffle:
            rng.shuffle(y_counts_per_group)
        sorted_groups_idx = np.argsort(-np.std(y_counts_per_group, axis=1), kind='mergesort')
        for group_idx in sorted_groups_idx:
            group_y_counts = y_counts_per_group[group_idx]
            best_fold = self._find_best_fold(y_counts_per_fold=y_counts_per_fold, y_cnt=y_cnt, group_y_counts=group_y_counts)
            y_counts_per_fold[best_fold] += group_y_counts
            groups_per_fold[best_fold].add(group_idx)
        for i in range(self.n_splits):
            test_indices = [idx for idx, group_idx in enumerate(groups_inv) if group_idx in groups_per_fold[i]]
            yield test_indices

    def _find_best_fold(self, y_counts_per_fold, y_cnt, group_y_counts):
        best_fold = None
        min_eval = np.inf
        min_samples_in_fold = np.inf
        for i in range(self.n_splits):
            y_counts_per_fold[i] += group_y_counts
            std_per_class = np.std(y_counts_per_fold / y_cnt.reshape(1, -1), axis=0)
            y_counts_per_fold[i] -= group_y_counts
            fold_eval = np.mean(std_per_class)
            samples_in_fold = np.sum(y_counts_per_fold[i])
            is_current_fold_better = fold_eval < min_eval or (np.isclose(fold_eval, min_eval) and samples_in_fold < min_samples_in_fold)
            if is_current_fold_better:
                min_eval = fold_eval
                min_samples_in_fold = samples_in_fold
                best_fold = i
        return best_fold