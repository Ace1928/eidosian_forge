from statsmodels.compat.pandas import assert_series_equal
from io import BytesIO
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
class TestRemoveDataPickleNegativeBinomial(RemoveDataPickle):

    def setup_method(self):
        np.random.seed(987689)
        data = sm.datasets.randhie.load()
        mod = sm.NegativeBinomial(data.endog, data.exog)
        self.results = mod.fit(disp=0)