from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.core.resample import Resampler as pd_Resampler
from dask.base import tokenize
from dask.dataframe import methods
from dask.dataframe._compat import PANDAS_GE_140
from dask.dataframe.core import DataFrame, Series
from dask.highlevelgraph import HighLevelGraph
from dask.utils import derived_from
def _resample_bin_and_out_divs(divisions, rule, closed='left', label='left'):
    rule = pd.tseries.frequencies.to_offset(rule)
    g = pd.Grouper(freq=rule, how='count', closed=closed, label=label)
    divs = pd.Series(range(len(divisions)), index=divisions)
    temp = divs.resample(rule, closed=closed, label='left').count()
    tempdivs = temp.loc[temp > 0].index
    res = pd.offsets.Nano() if isinstance(rule, pd.offsets.Tick) else pd.offsets.Day()
    if g.closed == 'right':
        newdivs = tempdivs + res
    else:
        newdivs = tempdivs
    if g.label == 'right':
        outdivs = tempdivs + rule
    else:
        outdivs = tempdivs
    newdivs = methods.tolist(newdivs)
    outdivs = methods.tolist(outdivs)
    if newdivs[0] < divisions[0]:
        newdivs[0] = divisions[0]
    if newdivs[-1] < divisions[-1]:
        if len(newdivs) < len(divs):
            setter = lambda a, val: a.append(val)
        else:
            setter = lambda a, val: a.__setitem__(-1, val)
        setter(newdivs, divisions[-1] + res)
        if outdivs[-1] > divisions[-1]:
            setter(outdivs, outdivs[-1])
        elif outdivs[-1] < divisions[-1]:
            setter(outdivs, temp.index[-1])
    return (tuple(map(pd.Timestamp, newdivs)), tuple(map(pd.Timestamp, outdivs)))