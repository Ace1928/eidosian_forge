import importlib
from collections import namedtuple
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn._config import config_context, get_config
from sklearn.preprocessing import StandardScaler
from sklearn.utils._set_output import (
from sklearn.utils.fixes import CSR_CONTAINERS
class BothMixinEstimator(_SetOutputMixin, AnotherMixin, custom_parameter=123):

    def transform(self, X, y=None):
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features