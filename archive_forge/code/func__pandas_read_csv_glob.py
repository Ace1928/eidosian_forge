import contextlib
import json
from pathlib import Path
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean
import modin.experimental.pandas as pd
from modin.config import AsyncReadMode, Engine
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import try_cast_to_pandas
def _pandas_read_csv_glob(path, storage_options):
    pandas_df = pandas.concat([pandas.read_csv(f'{s3_path}test_data{i}.csv', storage_options=storage_options) for i in range(2)]).reset_index(drop=True)
    return pandas_df