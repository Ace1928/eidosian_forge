from __future__ import print_function
import logging
import time as _time
import traceback
import multitasking as _multitasking
import pandas as _pd
from . import Ticker, utils
from .data import YfData
from . import shared
def _realign_dfs():
    idx_len = 0
    idx = None
    for df in shared._DFS.values():
        if len(df) > idx_len:
            idx_len = len(df)
            idx = df.index
    for key in shared._DFS.keys():
        try:
            shared._DFS[key] = _pd.DataFrame(index=idx, data=shared._DFS[key]).drop_duplicates()
        except Exception:
            shared._DFS[key] = _pd.concat([utils.empty_df(idx), shared._DFS[key].dropna()], axis=0, sort=True)
        shared._DFS[key] = shared._DFS[key].loc[~shared._DFS[key].index.duplicated(keep='last')]