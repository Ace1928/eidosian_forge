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
@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestSpss:

    def test_read_spss(self):
        test_args = ('fake_path',)
        test_kwargs = dict(usecols=['A'], convert_categoricals=False, dtype_backend=lib.no_default)
        with mock.patch('pandas.read_spss', return_value=pandas.DataFrame([])) as read_spss:
            pd.read_spss(*test_args, **test_kwargs)
        read_spss.assert_called_once_with(*test_args, **test_kwargs)