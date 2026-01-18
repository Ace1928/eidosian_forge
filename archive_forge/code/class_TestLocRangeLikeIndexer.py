import os
import sys
import matplotlib
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions, StorageFormat
from modin.pandas.indexing import is_range_like
from modin.pandas.testing import assert_index_equal
from modin.tests.pandas.utils import (
from modin.utils import get_current_execution
@pytest.mark.parametrize('indexer_size', [1, 2, NROWS, pytest.param(NROWS + 1, marks=pytest.mark.xfail(reason='https://github.com/modin-project/modin/issues/5739', strict=True))])
class TestLocRangeLikeIndexer:
    """Test cases related to https://github.com/modin-project/modin/issues/5702"""

    def test_range_index_getitem_single_value(self, indexer_size):
        eval_general(*create_test_dfs(test_data['int_data']), lambda df: df.loc[pd.RangeIndex(indexer_size)])

    def test_range_index_getitem_two_values(self, indexer_size):
        eval_general(*create_test_dfs(test_data['int_data']), lambda df: df.loc[pd.RangeIndex(indexer_size), :])

    def test_range_getitem_single_value(self, indexer_size):
        eval_general(*create_test_dfs(test_data['int_data']), lambda df: df.loc[range(indexer_size)])

    def test_range_getitem_two_values_5702(self, indexer_size):
        eval_general(*create_test_dfs(test_data['int_data']), lambda df: df.loc[range(indexer_size), :])