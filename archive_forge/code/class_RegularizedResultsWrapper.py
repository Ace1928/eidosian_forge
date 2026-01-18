import numpy as np
from statsmodels.base.model import Results
import statsmodels.base.wrapper as wrap
from statsmodels.tools.decorators import cache_readonly
class RegularizedResultsWrapper(wrap.ResultsWrapper):
    _attrs = {'params': 'columns', 'resid': 'rows', 'fittedvalues': 'rows'}
    _wrap_attrs = _attrs