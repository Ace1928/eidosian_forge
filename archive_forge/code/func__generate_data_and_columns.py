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
def _generate_data_and_columns(num_of_columns, num_of_records):
    data = []
    column_names = []
    for i in range(num_of_columns):
        data.append(_generate_random_int_array(size=num_of_records, min=1, max=num_of_records))
        column_names.append('c' + str(i))
    record_batch = pa.record_batch(data=data, names=column_names)
    return record_batch