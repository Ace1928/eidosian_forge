import io
import os
import sys
import tempfile
import pytest
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
from pyarrow.feather import (read_feather, write_feather, read_table,
def _check_arrow_roundtrip(table, path=None, compression=None):
    if path is None:
        path = random_path()
    TEST_FILES.append(path)
    write_feather(table, path, compression=compression)
    if not os.path.exists(path):
        raise Exception('file not written')
    result = read_table(path)
    assert result.equals(table)