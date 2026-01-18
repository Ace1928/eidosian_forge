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
def _test_read_parquet(self, engine, make_parquet_file, columns, filters, row_group_size, path_type=str, range_index_start=0, range_index_step=1, range_index_name=None, expected_exception=None):
    if engine == 'pyarrow' and filters == [] and (os.name == 'nt'):
        pytest.xfail('Skipping empty filters error case to avoid race condition - see #6460')
    with ensure_clean('.parquet') as unique_filename:
        unique_filename = path_type(unique_filename)
        make_parquet_file(filename=unique_filename, row_group_size=row_group_size, range_index_start=range_index_start, range_index_step=range_index_step, range_index_name=range_index_name)
        eval_io(fn_name='read_parquet', engine=engine, path=unique_filename, columns=columns, filters=filters, expected_exception=expected_exception)