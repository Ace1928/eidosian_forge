from scipy import stats
from scipy.stats import distributions
import numpy as np
class SquareFunc:
    """class to hold quadratic function with inverse function and derivative

    using instance methods instead of class methods, if we want extension
    to parametrized function
    """

    def inverseplus(self, x):
        return np.sqrt(x)

    def inverseminus(self, x):
        return 0.0 - np.sqrt(x)

    def derivplus(self, x):
        return 0.5 / np.sqrt(x)

    def derivminus(self, x):
        return 0.0 - 0.5 / np.sqrt(x)

    def squarefunc(self, x):
        return np.power(x, 2)