from statsmodels.compat.pandas import FUTURE_STACK
from statsmodels.compat.python import lzip
import datetime
from functools import reduce
import re
import textwrap
import numpy as np
import pandas as pd
from .table import SimpleTable
from .tableformatting import fmt_latex, fmt_txt
def add_base(self, results, alpha=0.05, float_format='%.4f', title=None, xname=None, yname=None):
    """Try to construct a basic summary instance.

        Parameters
        ----------
        results : Model results instance
        alpha : float
            significance level for the confidence intervals (optional)
        float_format: str
            Float formatting for summary of parameters (optional)
        title : str
            Title of the summary table (optional)
        xname : list[str] of length equal to the number of parameters
            Names of the independent variables (optional)
        yname : str
            Name of the dependent variable (optional)
        """
    param = summary_params(results, alpha=alpha, use_t=results.use_t)
    info = summary_model(results)
    if xname is not None:
        param.index = xname
    if yname is not None:
        info['Dependent Variable:'] = yname
    self.add_dict(info, align='l')
    self.add_df(param, float_format=float_format)
    self.add_title(title=title, results=results)