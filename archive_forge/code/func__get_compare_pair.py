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
def _get_compare_pair(data_source, record_batch, file_format, col_id):
    num_of_files_generated = _get_num_of_files_generated(base_directory=data_source, file_format=file_format)
    number_of_partitions = len(pa.compute.unique(record_batch[col_id]))
    return (num_of_files_generated, number_of_partitions)