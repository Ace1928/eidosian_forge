from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
import statsmodels.tools.data as data_util
from pandas import Index, MultiIndex
def _make_hierarchical_index(index, names):
    return MultiIndex.from_tuples(*[index], names=names)