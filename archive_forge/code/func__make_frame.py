import csv
from io import StringIO
import os
import numpy as np
import pytest
from pandas.errors import ParserError
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle
def _make_frame(names=None):
    if names is True:
        names = ['first', 'second']
    return DataFrame(np.random.default_rng(2).integers(0, 10, size=(3, 3)), columns=MultiIndex.from_tuples([('bah', 'foo'), ('bah', 'bar'), ('ban', 'baz')], names=names), dtype='int64')