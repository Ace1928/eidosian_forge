import warnings
from statsmodels.compat.pandas import Appender
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy import stats
from statsmodels.base.model import (
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
def _initialize_labels(self, labels, k_levels=None):
    self.labels = labels
    if k_levels is None:
        self.k_levels = len(labels)
    else:
        self.k_levels = k_levels
    if self.exog is not None:
        self.nobs, self.k_vars = self.exog.shape
    else:
        self.nobs, self.k_vars = (self.endog.shape[0], 0)
    threshold_names = [str(x) + '/' + str(y) for x, y in zip(labels[:-1], labels[1:])]
    if self.exog is not None:
        if len(self.exog_names) > self.k_vars:
            raise RuntimeError('something wrong with exog_names, too long')
        self.exog_names.extend(threshold_names)
    else:
        self.data.xnames = threshold_names