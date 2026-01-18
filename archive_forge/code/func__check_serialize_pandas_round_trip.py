from collections import UserList
import io
import pathlib
import pytest
import socket
import threading
import weakref
import numpy as np
import pyarrow as pa
from pyarrow.tests.util import changed_environ, invoke_script
@pytest.mark.pandas
def _check_serialize_pandas_round_trip(df, use_threads=False):
    buf = pa.serialize_pandas(df, nthreads=2 if use_threads else 1)
    result = pa.deserialize_pandas(buf, use_threads=use_threads)
    assert_frame_equal(result, df)