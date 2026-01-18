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
def _check_dataset(schema, expected, expected_schema=None):
    dataset = ds.dataset(str(tempdir / 'data.parquet'), schema=schema)
    if expected_schema is not None:
        assert dataset.schema.equals(expected_schema)
    else:
        assert dataset.schema.equals(schema)
    result = dataset_reader.to_table(dataset)
    assert result.equals(expected)