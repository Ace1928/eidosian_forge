from statsmodels.compat.python import lrange
import numpy as np
from itertools import combinations
class LeaveOneLabelOut:
    """
    Leave-One-Label_Out cross-validation iterator:
    Provides train/test indexes to split data in train test sets
    """

    def __init__(self, labels):
        """
        Leave-One-Label_Out cross validation:
        Provides train/test indexes to split data in train test sets

        Parameters
        ----------
        labels : list
                List of labels

        Examples
        --------
        >>> from scikits.learn import cross_val
        >>> X = [[1, 2], [3, 4], [5, 6], [7, 8]]
        >>> y = [1, 2, 1, 2]
        >>> labels = [1, 1, 2, 2]
        >>> lol = cross_val.LeaveOneLabelOut(labels)
        >>> for train_index, test_index in lol:
        ...    print "TRAIN:", train_index, "TEST:", test_index
        ...    X_train, X_test, y_train, y_test = cross_val.split(train_index,             test_index, X, y)
        ...    print X_train, X_test, y_train, y_test
        TRAIN: [False False  True  True] TEST: [ True  True False False]
        [[5 6]
        [7 8]] [[1 2]
        [3 4]] [1 2] [1 2]
        TRAIN: [ True  True False False] TEST: [False False  True  True]
        [[1 2]
        [3 4]] [[5 6]
        [7 8]] [1 2] [1 2]
        """
        self.labels = labels

    def __iter__(self):
        labels = np.array(self.labels, copy=True)
        for i in np.unique(labels):
            test_index = np.zeros(len(labels), dtype=bool)
            test_index[labels == i] = True
            train_index = np.logical_not(test_index)
            yield (train_index, test_index)

    def __repr__(self):
        return '{}.{}(labels={})'.format(self.__class__.__module__, self.__class__.__name__, self.labels)