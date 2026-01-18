import warnings
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lzip
from collections import defaultdict
import numpy as np
from statsmodels.graphics._regressionplots_doc import _plot_influence_doc
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import maybe_unwrap_results
@cache_readonly
def _get_prediction(self):
    with warnings.catch_warnings():
        msg = 'linear keyword is deprecated, use which="linear"'
        warnings.filterwarnings('ignore', message=msg, category=FutureWarning)
        pred = self.results.get_prediction()
    return pred