from statsmodels.compat.pandas import deprecate_kwarg
import os
import subprocess
import tempfile
import re
from warnings import warn
import pandas as pd
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (X13NotFoundError,
def _convert_out_to_series(x, dates, name):
    """
    Convert x to a DataFrame where x is a string in the format given by
    x-13arima-seats output.
    """
    from io import StringIO
    from pandas import read_csv
    out = read_csv(StringIO(x), skiprows=2, header=None, sep='\t', engine='python')
    return out.set_index(dates).rename(columns={1: name})[name]