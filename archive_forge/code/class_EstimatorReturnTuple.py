import importlib
from collections import namedtuple
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn._config import config_context, get_config
from sklearn.preprocessing import StandardScaler
from sklearn.utils._set_output import (
from sklearn.utils.fixes import CSR_CONTAINERS
class EstimatorReturnTuple(_SetOutputMixin):

    def __init__(self, OutputTuple):
        self.OutputTuple = OutputTuple

    def transform(self, X, y=None):
        return self.OutputTuple(X, 2 * X)