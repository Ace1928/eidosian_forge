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
def _simple_tables(tables, settings, pad_col=None, pad_index=None):
    simple_tables = []
    float_format = settings[0]['float_format'] if settings else '%.4f'
    if pad_col is None:
        pad_col = [0] * len(tables)
    if pad_index is None:
        pad_index = [0] * len(tables)
    for i, v in enumerate(tables):
        index = settings[i]['index']
        header = settings[i]['header']
        align = settings[i]['align']
        simple_tables.append(_df_to_simpletable(v, align=align, float_format=float_format, header=header, index=index, pad_col=pad_col[i], pad_index=pad_index[i]))
    return simple_tables