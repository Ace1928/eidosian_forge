from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
import statsmodels.tools.data as data_util
from pandas import Index, MultiIndex
def _make_generic_names(index):
    n_names = len(index.names)
    pad = str(len(str(n_names)))
    return [('group{0:0' + pad + '}').format(i) for i in range(n_names)]