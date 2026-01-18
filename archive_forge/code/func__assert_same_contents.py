import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def _assert_same_contents(join_chunk, source):
    NA_SENTINEL = -1234567
    jvalues = join_chunk.fillna(NA_SENTINEL).drop_duplicates().values
    svalues = source.fillna(NA_SENTINEL).drop_duplicates().values
    rows = {tuple(row) for row in jvalues}
    assert len(rows) == len(source)
    assert all((tuple(row) in rows for row in svalues))