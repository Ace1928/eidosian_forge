from warnings import warn
import numpy as np
from statsmodels.compat.pandas import Appender
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import OutputWarning, SpecificationWarning
import statsmodels.base.wrapper as wrap
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.tsatools import lagmat
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .initialization import Initialization
from .tools import (
def _validate_can_fix_params(self, param_names):
    super()._validate_can_fix_params(param_names)
    if 'ar_coeff' in self.parameters:
        ar_names = ['ar.L%d' % (i + 1) for i in range(self.ar_order)]
        fix_all_ar = param_names.issuperset(ar_names)
        fix_any_ar = len(param_names.intersection(ar_names)) > 0
        if fix_any_ar and (not fix_all_ar):
            raise ValueError('Cannot fix individual autoregressive. parameters. Must either fix all autoregressive parameters or none.')