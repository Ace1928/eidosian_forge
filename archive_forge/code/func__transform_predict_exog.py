from __future__ import annotations
from statsmodels.compat.python import lzip
from functools import reduce
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import handle_data
from statsmodels.base.optimizer import Optimizer
import statsmodels.base.wrapper as wrap
from statsmodels.formula import handle_formula_data
from statsmodels.stats.contrast import (
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import (
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import nan_dot, recipr
from statsmodels.tools.validation import bool_like
def _transform_predict_exog(self, exog, transform=True):
    is_pandas = _is_using_pandas(exog, None)
    exog_index = None
    if is_pandas:
        if exog.ndim == 2 or self.params.size == 1:
            exog_index = exog.index
        else:
            exog_index = [exog.index.name]
    if transform and hasattr(self.model, 'formula') and (exog is not None):
        design_info = getattr(self.model, 'design_info', None) or self.model.data.design_info
        from patsy import dmatrix
        if isinstance(exog, pd.Series):
            if hasattr(exog, 'name') and isinstance(exog.name, str) and (exog.name in design_info.describe()):
                exog = pd.DataFrame(exog)
            else:
                exog = pd.DataFrame(exog).T
            exog_index = exog.index
        orig_exog_len = len(exog)
        is_dict = isinstance(exog, dict)
        try:
            exog = dmatrix(design_info, exog, return_type='dataframe')
        except Exception as exc:
            msg = 'predict requires that you use a DataFrame when predicting from a model\nthat was created using the formula api.\n\nThe original error message returned by patsy is:\n{}'.format(str(str(exc)))
            raise exc.__class__(msg)
        if orig_exog_len > len(exog) and (not is_dict):
            if exog_index is None:
                warnings.warn('nan values have been dropped', ValueWarning)
            else:
                exog = exog.reindex(exog_index)
        exog_index = exog.index
    if exog is not None:
        exog = np.asarray(exog)
        if exog.ndim == 1 and (self.model.exog.ndim == 1 or self.model.exog.shape[1] == 1):
            exog = exog[:, None]
        exog = np.atleast_2d(exog)
    return (exog, exog_index)