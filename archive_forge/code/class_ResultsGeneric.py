import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
class ResultsGeneric:

    def __init__(self, **kwds):
        self.__dict__.update(kwds)