import math as _math
from abc import ABCMeta, abstractmethod
from functools import reduce
@staticmethod
def _expected_values(cont):
    """Calculates expected values for a contingency table."""
    n_xx = sum(cont)
    for i in range(4):
        yield ((cont[i] + cont[i ^ 1]) * (cont[i] + cont[i ^ 2]) / n_xx)