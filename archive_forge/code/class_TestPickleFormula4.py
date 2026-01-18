from statsmodels.compat.pandas import assert_series_equal
from io import BytesIO
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
class TestPickleFormula4(TestPickleFormula2):

    def setup_method(self):
        self.results = sm.OLS.from_formula('Y ~ np.log(abs(A) + 1) + B * C', data=self.data).fit()