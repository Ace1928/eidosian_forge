import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Optional
import boto3
import numpy as np
import pandas
import pytest
import requests
import s3fs
from pandas.util._decorators import doc
import modin.utils  # noqa: E402
import uuid  # noqa: E402
import modin  # noqa: E402
import modin.config  # noqa: E402
import modin.tests.config  # noqa: E402
from modin.config import (  # noqa: E402
from modin.core.execution.dispatching.factories import factories  # noqa: E402
from modin.core.execution.python.implementations.pandas_on_python.io import (  # noqa: E402
from modin.core.storage_formats import (  # noqa: E402
from modin.tests.pandas.utils import (  # noqa: E402
def _make_parquet_file(filename, nrows=NROWS, ncols=2, force=True, range_index_start=0, range_index_step=1, range_index_name=None, partitioned_columns=[], row_group_size: Optional[int]=None):
    """Helper function to generate parquet files/directories.

        Args:
            filename: The name of test file, that should be created.
            nrows: Number of rows for the dataframe.
            ncols: Number of cols for the dataframe.
            force: Create a new file/directory even if one already exists.
            partitioned_columns: Create a partitioned directory using pandas.
            row_group_size: Maximum size of each row group.
        """
    if force or not os.path.exists(filename):
        df = pandas.DataFrame({f'col{x + 1}': np.arange(nrows) for x in range(ncols)})
        index = pandas.RangeIndex(start=range_index_start, stop=range_index_start + nrows * range_index_step, step=range_index_step, name=range_index_name)
        if range_index_start == 0 and range_index_step == 1 and (range_index_name is None):
            assert df.index.equals(index)
        else:
            df.index = index
        if len(partitioned_columns) > 0:
            df.to_parquet(filename, partition_cols=partitioned_columns, row_group_size=row_group_size)
        else:
            df.to_parquet(filename, row_group_size=row_group_size)
        filenames.append(filename)