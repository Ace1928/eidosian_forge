from statsmodels.compat.pandas import assert_series_equal
from io import BytesIO
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
class TestRemoveDataPicklePoisson(RemoveDataPickle):

    def setup_method(self):
        x = self.exog
        np.random.seed(987689)
        y_count = np.random.poisson(np.exp(x.sum(1) - x.mean()))
        model = sm.Poisson(y_count, x)
        start_params = np.array([0.75334818, 0.99425553, 1.00494724, 1.00247112])
        self.results = model.fit(start_params=start_params, method='bfgs', disp=0)
        self.predict_kwds = dict(exposure=1, offset=0)