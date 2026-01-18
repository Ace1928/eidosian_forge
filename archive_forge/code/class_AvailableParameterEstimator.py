import pickle
import pytest
from sklearn.utils.metaestimators import available_if
class AvailableParameterEstimator:
    """This estimator's `available` parameter toggles the presence of a method"""

    def __init__(self, available=True, return_value=1):
        self.available = available
        self.return_value = return_value

    @available_if(lambda est: est.available)
    def available_func(self):
        """This is a mock available_if function"""
        return self.return_value