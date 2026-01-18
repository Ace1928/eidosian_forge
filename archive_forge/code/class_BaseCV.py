from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import itertools
import numpy as np
from statsmodels.gam.smooth_basis import (GenericSmoothers,
class BaseCV(with_metaclass(ABCMeta)):
    """
    BaseCV class. It computes the cross validation error of a given model.
    All the cross validation classes can be derived by this one
    (e.g. GamCV, LassoCV,...)
    """

    def __init__(self, cv_iterator, endog, exog):
        self.cv_iterator = cv_iterator
        self.exog = exog
        self.endog = endog
        self.train_test_cv_indices = self.cv_iterator.split(self.exog, self.endog, label=None)

    def fit(self, **kwargs):
        cv_err = []
        for train_index, test_index in self.train_test_cv_indices:
            cv_err.append(self._error(train_index, test_index, **kwargs))
        return np.array(cv_err)

    @abstractmethod
    def _error(self, train_index, test_index, **kwargs):
        pass