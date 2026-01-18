from statsmodels.compat.pandas import assert_series_equal
from io import BytesIO
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
class TestPickleFormula2(RemoveDataPickle):

    @classmethod
    def setup_class(cls):
        super().setup_class()
        nobs = 500
        np.random.seed(987689)
        data = np.random.randn(nobs, 4)
        data[:, 0] = data[:, 1:].sum(1)
        cls.data = pd.DataFrame(data, columns=['Y', 'A', 'B', 'C'])
        cls.xf = pd.DataFrame(0.25 * np.ones((2, 3)), columns=cls.data.columns[1:])
        cls.reduction_factor = 0.5

    def setup_method(self):
        self.results = sm.OLS.from_formula('Y ~ A + B + C', data=self.data).fit()