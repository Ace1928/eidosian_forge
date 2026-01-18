import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.stats.base import HolderTuple
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.regression.linear_model import OLS
class DispersionResults(HolderTuple):

    def summary_frame(self):
        frame = pd.DataFrame({'statistic': self.statistic, 'pvalue': self.pvalue, 'method': self.method, 'alternative': self.alternative})
        return frame