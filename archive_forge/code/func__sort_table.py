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
def _sort_table(tab, sort_col):
    import pyarrow.compute as pc
    sorted_indices = pc.sort_indices(tab, options=pc.SortOptions([(sort_col, 'ascending')]))
    return pc.take(tab, sorted_indices)