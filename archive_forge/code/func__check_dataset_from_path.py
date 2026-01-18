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
def _check_dataset_from_path(path, table, dataset_reader, pickler, **kwargs):
    assert isinstance(path, pathlib.Path)
    for p in [path, str(path), [path], [str(path)]]:
        dataset = ds.dataset(path, **kwargs)
        assert isinstance(dataset, ds.FileSystemDataset)
        _check_dataset(dataset, table, dataset_reader, pickler)
    with change_cwd(path.parent):
        dataset = ds.dataset(path.name, **kwargs)
        assert isinstance(dataset, ds.FileSystemDataset)
        _check_dataset(dataset, table, dataset_reader, pickler)