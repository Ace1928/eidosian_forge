from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import itertools
import numpy as np
from statsmodels.gam.smooth_basis import (GenericSmoothers,
class BasePenaltiesPathCV(with_metaclass(ABCMeta)):
    """
    Base class for cross validation over a grid of parameters.

    The best parameter is saved in alpha_cv

    This class is currently not used
    """

    def __init__(self, alphas):
        self.alphas = alphas
        self.alpha_cv = None
        self.cv_error = None
        self.cv_std = None

    def plot_path(self):
        from statsmodels.graphics.utils import _import_mpl
        plt = _import_mpl()
        plt.plot(self.alphas, self.cv_error, c='black')
        plt.plot(self.alphas, self.cv_error + 1.96 * self.cv_std, c='blue')
        plt.plot(self.alphas, self.cv_error - 1.96 * self.cv_std, c='blue')
        plt.plot(self.alphas, self.cv_error, 'o', c='black')
        plt.plot(self.alphas, self.cv_error + 1.96 * self.cv_std, 'o', c='blue')
        plt.plot(self.alphas, self.cv_error - 1.96 * self.cv_std, 'o', c='blue')
        return