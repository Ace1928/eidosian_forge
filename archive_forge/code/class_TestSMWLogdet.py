from statsmodels.compat.platform import PLATFORM_OSX
import os
import csv
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
import pytest
from statsmodels.regression.mixed_linear_model import (
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from statsmodels.base import _penalties as penalties
import statsmodels.tools.numdiff as nd
from .results import lme_r_results
class TestSMWLogdet:

    @classmethod
    def setup_class(cls):
        np.random.seed(23)

    @pytest.mark.parametrize('p', [5, 10])
    @pytest.mark.parametrize('q', [4, 8])
    @pytest.mark.parametrize('r', [2, 3])
    @pytest.mark.parametrize('s', [0, 0.5])
    def test_smw_logdet(self, p, q, r, s):
        check_smw_logdet(p, q, r, s)