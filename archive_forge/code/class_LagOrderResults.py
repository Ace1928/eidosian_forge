from __future__ import annotations
from statsmodels.compat.python import lrange
from collections import defaultdict
from io import StringIO
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly, deprecated_alias
from statsmodels.tools.linalg import logdet_symm
from statsmodels.tools.sm_exceptions import OutputWarning
from statsmodels.tools.validation import array_like
from statsmodels.tsa.base.tsa_model import (
import statsmodels.tsa.tsatools as tsa
from statsmodels.tsa.tsatools import duplication_matrix, unvec, vec
from statsmodels.tsa.vector_ar import output, plotting, util
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.output import VARSummary
class LagOrderResults:
    """
    Results class for choosing a model's lag order.

    Parameters
    ----------
    ics : dict
        The keys are the strings ``"aic"``, ``"bic"``, ``"hqic"``, and
        ``"fpe"``. A corresponding value is a list of information criteria for
        various numbers of lags.
    selected_orders : dict
        The keys are the strings ``"aic"``, ``"bic"``, ``"hqic"``, and
        ``"fpe"``. The corresponding value is an integer specifying the number
        of lags chosen according to a given criterion (key).
    vecm : bool, default: `False`
        `True` indicates that the model is a VECM. In case of a VAR model
        this argument must be `False`.

    Notes
    -----
    In case of a VECM the shown lags are lagged differences.
    """

    def __init__(self, ics, selected_orders, vecm=False):
        self.title = ('VECM' if vecm else 'VAR') + ' Order Selection'
        self.title += ' (* highlights the minimums)'
        self.ics = ics
        self.selected_orders = selected_orders
        self.vecm = vecm
        self.aic = selected_orders['aic']
        self.bic = selected_orders['bic']
        self.hqic = selected_orders['hqic']
        self.fpe = selected_orders['fpe']

    def summary(self):
        cols = sorted(self.ics)
        str_data = np.array([['%#10.4g' % v for v in self.ics[c]] for c in cols], dtype=object).T
        for i, col in enumerate(cols):
            idx = (int(self.selected_orders[col]), i)
            str_data[idx] += '*'
        return SimpleTable(str_data, [col.upper() for col in cols], lrange(len(str_data)), title=self.title)

    def __str__(self):
        return f'<{self.__module__}.{self.__class__.__name__} object. Selected orders are: AIC -> {str(self.aic)}, BIC -> {str(self.bic)}, FPE -> {str(self.fpe)}, HQIC ->  {str(self.hqic)}>'