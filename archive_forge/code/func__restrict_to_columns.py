import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def _restrict_to_columns(group, columns, suffix):
    found = [c for c in group.columns if c in columns or c.replace(suffix, '') in columns]
    group = group.loc[:, found]
    group = group.rename(columns=lambda x: x.replace(suffix, ''))
    group = group.loc[:, columns]
    return group