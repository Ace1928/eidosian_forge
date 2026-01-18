import warnings
from statsmodels.compat.pandas import PD_LT_1_4
import os
import numpy as np
import pandas as pd
from statsmodels.multivariate.factor import Factor
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
def _zscore(x):
    return (x - x.mean(0)) / x.std(0)