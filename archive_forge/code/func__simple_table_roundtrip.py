import decimal
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _check_roundtrip
def _simple_table_roundtrip(table, **write_kwargs):
    stream = pa.BufferOutputStream()
    _write_table(table, stream, **write_kwargs)
    buf = stream.getvalue()
    return _read_table(buf)