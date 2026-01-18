import contextlib
from warnings import warn
import pandas as pd
import numpy as np
from statsmodels.compat.pandas import Appender
from statsmodels.tools.tools import Bunch
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tsa.vector_ar import var_model
import statsmodels.base.wrapper as wrap
from statsmodels.tools.sm_exceptions import EstimationWarning
from .kalman_filter import INVERT_UNIVARIATE, SOLVE_LU
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
from .initialization import Initialization
from .tools import (
def _news_previous_results(self, previous, start, end, periods, revisions_details_start=False, state_index=None):
    exog = None
    out_of_sample = self.nobs - previous.nobs
    if self.model.k_exog > 0 and out_of_sample > 0:
        exog = self.model.exog[-out_of_sample:]
    with contextlib.ExitStack() as stack:
        stack.enter_context(previous.model._set_final_exog(exog))
        stack.enter_context(previous._set_final_predicted_state(exog, out_of_sample))
        out = self.smoother_results.news(previous.smoother_results, start=start, end=end, revisions_details_start=revisions_details_start, state_index=state_index)
    return out