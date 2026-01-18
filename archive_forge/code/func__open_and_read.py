from statsmodels.compat.pandas import deprecate_kwarg
import os
import subprocess
import tempfile
import re
from warnings import warn
import pandas as pd
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (X13NotFoundError,
def _open_and_read(fname):
    with open(fname, encoding='utf-8') as fin:
        fout = fin.read()
    return fout