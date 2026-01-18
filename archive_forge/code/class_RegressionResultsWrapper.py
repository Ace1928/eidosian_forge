from __future__ import annotations
from statsmodels.compat.pandas import Appender
from statsmodels.compat.python import lrange, lzip
from typing import Literal
from collections.abc import Sequence
import warnings
import numpy as np
from scipy import optimize, stats
from scipy.linalg import cholesky, toeplitz
from scipy.linalg.lapack import dtrtri
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.emplike.elregress import _ELRegOpts
from statsmodels.regression._prediction import PredictionResults
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import pinv_extended
from statsmodels.tools.typing import Float64Array
from statsmodels.tools.validation import bool_like, float_like, string_like
from . import _prediction as pred
class RegressionResultsWrapper(wrap.ResultsWrapper):
    _attrs = {'chisq': 'columns', 'sresid': 'rows', 'weights': 'rows', 'wresid': 'rows', 'bcov_unscaled': 'cov', 'bcov_scaled': 'cov', 'HC0_se': 'columns', 'HC1_se': 'columns', 'HC2_se': 'columns', 'HC3_se': 'columns', 'norm_resid': 'rows'}
    _wrap_attrs = wrap.union_dicts(base.LikelihoodResultsWrapper._attrs, _attrs)
    _methods = {}
    _wrap_methods = wrap.union_dicts(base.LikelihoodResultsWrapper._wrap_methods, _methods)