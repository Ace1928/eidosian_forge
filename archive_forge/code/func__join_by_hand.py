import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def _join_by_hand(a, b, how='left'):
    join_index = a.index.join(b.index, how=how)
    a_re = a.reindex(join_index)
    b_re = b.reindex(join_index)
    result_columns = a.columns.append(b.columns)
    for col, s in b_re.items():
        a_re[col] = s
    return a_re.reindex(columns=result_columns)