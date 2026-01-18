import datetime
import inspect
import os
import pathlib
import numpy as np
import pytest
import unittest.mock as mock
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem
from pyarrow.tests import util
from pyarrow.util import guid
from pyarrow.vendored.version import Version
def _partition_test_for_filesystem(fs, base_path):
    foo_keys = [0, 1]
    bar_keys = ['a', 'b', 'c']
    partition_spec = [['foo', foo_keys], ['bar', bar_keys]]
    N = 30
    df = pd.DataFrame({'index': np.arange(N), 'foo': np.array(foo_keys, dtype='i4').repeat(15), 'bar': np.tile(np.tile(np.array(bar_keys, dtype=object), 5), 2), 'values': np.random.randn(N)}, columns=['index', 'foo', 'bar', 'values'])
    _generate_partition_directories(fs, base_path, partition_spec, df)
    dataset = pq.ParquetDataset(base_path, filesystem=fs)
    table = dataset.read()
    result_df = table.to_pandas().sort_values(by='index').reset_index(drop=True)
    expected_df = df.sort_values(by='index').reset_index(drop=True).reindex(columns=result_df.columns)
    expected_df['foo'] = expected_df['foo'].astype('category')
    expected_df['bar'] = expected_df['bar'].astype('category')
    assert (result_df.columns == ['index', 'values', 'foo', 'bar']).all()
    tm.assert_frame_equal(result_df, expected_df)