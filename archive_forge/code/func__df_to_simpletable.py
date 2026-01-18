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
def _df_to_simpletable(df, align='r', float_format='%.4f', header=True, index=True, table_dec_above='-', table_dec_below=None, header_dec_below='-', pad_col=0, pad_index=0):
    dat = df.copy()
    try:
        dat = dat.map(lambda x: _formatter(x, float_format))
    except AttributeError:
        dat = dat.applymap(lambda x: _formatter(x, float_format))
    if header:
        headers = [str(x) for x in dat.columns.tolist()]
    else:
        headers = None
    if index:
        stubs = [str(x) + int(pad_index) * ' ' for x in dat.index.tolist()]
    else:
        dat.iloc[:, 0] = [str(x) + int(pad_index) * ' ' for x in dat.iloc[:, 0]]
        stubs = None
    st = SimpleTable(np.array(dat), headers=headers, stubs=stubs, ltx_fmt=fmt_latex, txt_fmt=fmt_txt)
    st.output_formats['latex']['data_aligns'] = align
    st.output_formats['latex']['header_align'] = align
    st.output_formats['txt']['data_aligns'] = align
    st.output_formats['txt']['table_dec_above'] = table_dec_above
    st.output_formats['txt']['table_dec_below'] = table_dec_below
    st.output_formats['txt']['header_dec_below'] = header_dec_below
    st.output_formats['txt']['colsep'] = ' ' * int(pad_col + 1)
    return st