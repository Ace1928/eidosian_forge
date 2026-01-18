import contextlib
import csv
import inspect
import os
import sys
import unittest.mock as mock
from collections import defaultdict
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict
import fastparquet
import numpy as np
import pandas
import pandas._libs.lib as lib
import pyarrow as pa
import pyarrow.dataset
import pytest
import sqlalchemy as sa
from packaging import version
from pandas._testing import ensure_clean
from pandas.errors import ParserWarning
from scipy import sparse
from modin.config import (
from modin.db_conn import ModinDatabaseConnection, UnsupportedDatabaseException
from modin.pandas.io import from_arrow, from_dask, from_ray, to_pandas
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from .utils import (
from .utils import test_data as utils_test_data
from .utils import time_parsing_csv_path
from modin.config import NPartitions
def _test_read_parquet_directory(self, engine, make_parquet_dir, columns, filters, range_index_start, range_index_step, range_index_name, row_group_size, rows_per_file):
    num_cols = DATASET_SIZE_DICT.get(TestDatasetSize.get(), DATASET_SIZE_DICT['Small'])
    dfs_by_filename = {}
    start_row = 0
    for i, length in enumerate(rows_per_file):
        end_row = start_row + length
        df = pandas.DataFrame({f'col{x + 1}': np.arange(start_row, end_row) for x in range(num_cols)})
        index = pandas.RangeIndex(start=range_index_start, stop=range_index_start + length * range_index_step, step=range_index_step, name=range_index_name)
        if range_index_start == 0 and range_index_step == 1 and (range_index_name is None):
            assert df.index.equals(index)
        else:
            df.index = index
        dfs_by_filename[f'{i}.parquet'] = df
        start_row = end_row
    path = make_parquet_dir(dfs_by_filename, row_group_size)
    with open(os.path.join(path, '_committed_file'), 'w+') as f:
        f.write('testingtesting')
    eval_io(fn_name='read_parquet', engine=engine, path=path, columns=columns, filters=filters)