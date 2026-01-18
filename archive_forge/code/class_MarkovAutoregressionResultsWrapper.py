import numpy as np
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.regime_switching import (
from statsmodels.tsa.statespace.tools import (
class MarkovAutoregressionResultsWrapper(markov_regression.MarkovRegressionResultsWrapper):
    pass