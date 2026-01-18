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
def _test_write_to_dataset_no_partitions(base_path, filesystem=None):
    import pandas as pd
    import pyarrow.parquet as pq
    output_df = pd.DataFrame({'group1': list('aaabbbbccc'), 'group2': list('eefeffgeee'), 'num': list(range(10)), 'date': np.arange('2017-01-01', '2017-01-11', dtype='datetime64[D]').astype('datetime64[ns]')})
    cols = output_df.columns.tolist()
    output_table = pa.Table.from_pandas(output_df)
    if filesystem is None:
        filesystem = LocalFileSystem._get_instance()
    n = 5
    for i in range(n):
        pq.write_to_dataset(output_table, base_path, filesystem=filesystem)
    output_files = [file for file in filesystem.ls(str(base_path)) if file.endswith('.parquet')]
    assert len(output_files) == n
    input_table = pq.ParquetDataset(base_path, filesystem=filesystem).read()
    input_df = input_table.to_pandas()
    input_df = input_df.drop_duplicates()
    input_df = input_df[cols]
    tm.assert_frame_equal(output_df, input_df)