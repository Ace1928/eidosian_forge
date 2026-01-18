import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def _check_join(left, right, result, join_col, how='left', lsuffix='_x', rsuffix='_y'):
    for c in join_col:
        assert result[c].notna().all()
    left_grouped = left.groupby(join_col)
    right_grouped = right.groupby(join_col)
    for group_key, group in result.groupby(join_col):
        l_joined = _restrict_to_columns(group, left.columns, lsuffix)
        r_joined = _restrict_to_columns(group, right.columns, rsuffix)
        try:
            lgroup = left_grouped.get_group(group_key)
        except KeyError as err:
            if how in ('left', 'inner'):
                raise AssertionError(f'key {group_key} should not have been in the join') from err
            _assert_all_na(l_joined, left.columns, join_col)
        else:
            _assert_same_contents(l_joined, lgroup)
        try:
            rgroup = right_grouped.get_group(group_key)
        except KeyError as err:
            if how in ('right', 'inner'):
                raise AssertionError(f'key {group_key} should not have been in the join') from err
            _assert_all_na(r_joined, right.columns, join_col)
        else:
            _assert_same_contents(r_joined, rgroup)