from statsmodels.compat.pandas import MONTH_END
import os
import pickle
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from statsmodels.datasets import co2
from statsmodels.tsa.seasonal import STL, DecomposeResult
def _to_class_kwargs(kwargs, robust=False):
    endog = kwargs['y']
    np = kwargs['np']
    ns = kwargs['ns']
    nt = kwargs['nt']
    nl = kwargs['nl']
    isdeg = kwargs['isdeg']
    itdeg = kwargs['itdeg']
    ildeg = kwargs['ildeg']
    nsjump = kwargs['nsjump']
    ntjump = kwargs['ntjump']
    nljump = kwargs['nljump']
    outer_iter = kwargs['no']
    inner_iter = kwargs['ni']
    class_kwargs = dict(endog=endog, period=np, seasonal=ns, trend=nt, low_pass=nl, seasonal_deg=isdeg, trend_deg=itdeg, low_pass_deg=ildeg, robust=robust, seasonal_jump=nsjump, trend_jump=ntjump, low_pass_jump=nljump)
    return (class_kwargs, outer_iter, inner_iter)