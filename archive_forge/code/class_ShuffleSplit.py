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
class ShuffleSplit(BaseShuffleSplit):
    """Random permutation cross-validator.

    Yields indices to split data into training and test sets.

    Note: contrary to other cross-validation strategies, random splits
    do not guarantee that all folds will be different, although this is
    still very likely for sizeable datasets.

    Read more in the :ref:`User Guide <ShuffleSplit>`.

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
    >>> from sklearn.model_selection import ShuffleSplit
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [3, 4], [5, 6]])
    >>> y = np.array([1, 2, 1, 2, 1, 2])
    >>> rs = ShuffleSplit(n_splits=5, test_size=.25, random_state=0)
    >>> rs.get_n_splits(X)
    5
    >>> print(rs)
    ShuffleSplit(n_splits=5, random_state=0, test_size=0.25, train_size=None)
    >>> for i, (train_index, test_index) in enumerate(rs.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[1 3 0 4]
      Test:  index=[5 2]
    Fold 1:
      Train: index=[4 0 2 5]
      Test:  index=[1 3]
    Fold 2:
      Train: index=[1 2 4 0]
      Test:  index=[3 5]
    Fold 3:
      Train: index=[3 4 1 0]
      Test:  index=[5 2]
    Fold 4:
      Train: index=[3 5 1 0]
      Test:  index=[2 4]
    >>> # Specify train and test size
    >>> rs = ShuffleSplit(n_splits=5, train_size=0.5, test_size=.25,
    ...                   random_state=0)
    >>> for i, (train_index, test_index) in enumerate(rs.split(X)):
    ...     print(f"Fold {i}:")
    ...     print(f"  Train: index={train_index}")
    ...     print(f"  Test:  index={test_index}")
    Fold 0:
      Train: index=[1 3 0]
      Test:  index=[5 2]
    Fold 1:
      Train: index=[4 0 2]
      Test:  index=[1 3]
    Fold 2:
      Train: index=[1 2 4]
      Test:  index=[3 5]
    Fold 3:
      Train: index=[3 4 1]
      Test:  index=[5 2]
    Fold 4:
      Train: index=[3 5 1]
      Test:  index=[2 4]
    """

    def __init__(self, n_splits=10, *, test_size=None, train_size=None, random_state=None):
        super().__init__(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=random_state)
        self._default_test_size = 0.1

    def _iter_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        n_train, n_test = _validate_shuffle_split(n_samples, self.test_size, self.train_size, default_test_size=self._default_test_size)
        rng = check_random_state(self.random_state)
        for i in range(self.n_splits):
            permutation = rng.permutation(n_samples)
            ind_test = permutation[:n_test]
            ind_train = permutation[n_test:n_test + n_train]
            yield (ind_train, ind_test)