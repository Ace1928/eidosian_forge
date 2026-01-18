import warnings
import numpy as np
import pandas as pd
from statsmodels.base import model
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import ConvergenceWarning
class DimReductionResultsWrapper(wrap.ResultsWrapper):
    _attrs = {'params': 'columns'}
    _wrap_attrs = _attrs