import contextlib
import datetime
import os
import pathlib
import posixpath
import sys
import tempfile
import textwrap
import threading
import time
from shutil import copytree
from urllib.parse import quote
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
import pyarrow.json
from pyarrow.tests.util import (FSProtocolClass, ProxyHandler,
def _create_parquet_dataset_partitioned(root_path):
    table = pa.table([pa.array(range(20)), pa.array(np.random.randn(20)), pa.array(np.repeat(['a', 'b'], 10))], names=['f1', 'f2', 'part'])
    table = table.replace_schema_metadata({'key': 'value'})
    pq.write_to_dataset(table, str(root_path), partition_cols=['part'])
    return (_create_metadata_file(root_path), table)