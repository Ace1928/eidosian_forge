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
def _create_dataset_all_types(tempdir, chunk_size=None):
    table = pa.table([pa.array([True, None, False], pa.bool_()), pa.array([1, 10, 42], pa.int8()), pa.array([1, 10, 42], pa.uint8()), pa.array([1, 10, 42], pa.int16()), pa.array([1, 10, 42], pa.uint16()), pa.array([1, 10, 42], pa.int32()), pa.array([1, 10, 42], pa.uint32()), pa.array([1, 10, 42], pa.int64()), pa.array([1, 10, 42], pa.uint64()), pa.array([1.0, 10.0, 42.0], pa.float32()), pa.array([1.0, 10.0, 42.0], pa.float64()), pa.array(['a', None, 'z'], pa.utf8()), pa.array(['a', None, 'z'], pa.binary()), pa.array([1, 10, 42], pa.timestamp('s')), pa.array([1, 10, 42], pa.timestamp('ms')), pa.array([1, 10, 42], pa.timestamp('us')), pa.array([1, 10, 42], pa.date32()), pa.array([1, 10, 4200000000], pa.date64()), pa.array([1, 10, 42], pa.time32('s')), pa.array([1, 10, 42], pa.time64('us'))], names=['boolean', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float', 'double', 'utf8', 'binary', 'ts[s]', 'ts[ms]', 'ts[us]', 'date32', 'date64', 'time32', 'time64'])
    path = str(tempdir / 'test_parquet_dataset_all_types')
    pq.write_to_dataset(table, path, chunk_size=chunk_size)
    return (table, ds.dataset(path, format='parquet', partitioning='hive'))