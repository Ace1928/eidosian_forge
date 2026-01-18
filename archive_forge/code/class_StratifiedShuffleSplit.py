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
class StratifiedShuffleSplit(BaseShuffleSplit):
    """Stratified ShuffleSplit cross-validator.

    Provides train/test indices to split data in train/test sets.

    This cross-validation object is a merge of StratifiedKFold and
    ShuffleSplit, which returns stratified randomized folds. The folds
    are made by preserving the percentage of samples for each class.

    Note: like the ShuffleSplit strategy, stratified random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.

    Read more in the :ref:`User Guide <stratified_shuffle_split>`.

    For visualisation of cross-validation behaviour and
    comparison between common scikit-learn split methods
    refer to :ref:`sphx_glr_auto_examples_model_selection_plot_cv_indices.py`

    Parameters
    ----------
    n_splits : int, default=10
        Number of re-shuffling & splitting iterations.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.1.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import StratifiedShuffleSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    >>> sss.get_n_splits(X, y)
    5
    >>> print(sss)
    StratifiedShuffleSplit(n_splits=5, random_state=0, ...)
    >>> for i, (train_index, test_index) in enumerate(sss.split(X, y)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[5 2 3]
      Test:  index=[4 1 0]
    Fold 1:
      Train: index=[5 1 4]
      Test:  index=[0 2 3]
    Fold 2:
      Train: index=[5 0 2]
      Test:  index=[4 3 1]
    Fold 3:
      Train: index=[4 1 0]
      Test:  index=[2 3 5]
    Fold 4:
      Train: index=[0 5 1]
      Test:  index=[3 4 2]
    """

    def __init__(self, n_splits=10, *, test_size=None, train_size=None, random_state=None):
        super().__init__(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=random_state)
        self._default_test_size = 0.1

    def _iter_indices(self, X, y, groups=None):
        n_samples = _num_samples(X)
        y = check_array(y, input_name='y', ensure_2d=False, dtype=None)
        n_train, n_test = _validate_shuffle_split(n_samples, self.test_size, self.train_size, default_test_size=self._default_test_size)
        if y.ndim == 2:
            y = np.array([' '.join(row.astype('str')) for row in y])
        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = classes.shape[0]
        class_counts = np.bincount(y_indices)
        if np.min(class_counts) < 2:
            raise ValueError('The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.')
        if n_train < n_classes:
            raise ValueError('The train_size = %d should be greater or equal to the number of classes = %d' % (n_train, n_classes))
        if n_test < n_classes:
            raise ValueError('The test_size = %d should be greater or equal to the number of classes = %d' % (n_test, n_classes))
        class_indices = np.split(np.argsort(y_indices, kind='mergesort'), np.cumsum(class_counts)[:-1])
        rng = check_random_state(self.random_state)
        for _ in range(self.n_splits):
            n_i = _approximate_mode(class_counts, n_train, rng)
            class_counts_remaining = class_counts - n_i
            t_i = _approximate_mode(class_counts_remaining, n_test, rng)
            train = []
            test = []
            for i in range(n_classes):
                permutation = rng.permutation(class_counts[i])
                perm_indices_class_i = class_indices[i].take(permutation, mode='clip')
                train.extend(perm_indices_class_i[:n_i[i]])
                test.extend(perm_indices_class_i[n_i[i]:n_i[i] + t_i[i]])
            train = rng.permutation(train)
            test = rng.permutation(test)
            yield (train, test)

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            Note that providing ``y`` is sufficient to generate the splits and
            hence ``np.zeros(n_samples)`` may be used as a placeholder for
            ``X`` instead of actual training data.

        y : array-like of shape (n_samples,) or (n_samples, n_labels)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : object
            Always ignored, exists for compatibility.

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
        y = check_array(y, input_name='y', ensure_2d=False, dtype=None)
        return super().split(X, y, groups)