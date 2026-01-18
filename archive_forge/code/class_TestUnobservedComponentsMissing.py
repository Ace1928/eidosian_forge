import os
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from scipy.linalg import cho_solve_banded
from statsmodels import datasets
from statsmodels.tsa.statespace import (sarimax, structural, dynamic_factor,
class TestUnobservedComponentsMissing(TestUnobservedComponents):

    def setup_class(cls, missing='mixed', *args, **kwargs):
        super().setup_class(*args, missing=missing, **kwargs)