from __future__ import annotations
import copy
import pickle
import threading
import warnings
from collections import OrderedDict, defaultdict
from contextlib import ExitStack
import numpy as np
import pandas as pd
import tlz as toolz
from packaging.version import parse as parse_version
from dask.core import flatten
from dask.dataframe._compat import PANDAS_GE_201
from dask.base import tokenize
from dask.dataframe.io.parquet.utils import (
from dask.dataframe.io.utils import _is_local_fs, _meta_from_dtypes, _open_input_files
from dask.dataframe.utils import UNKNOWN_CATEGORIES
from dask.delayed import Delayed
from dask.utils import natural_sort_key
@classmethod
def _make_part(cls, filename, rg_list, fs=None, pf=None, base_path=None, partitions=None):
    """Generate a partition-specific element of `parts`."""
    if partitions:
        real_row_groups = cls._get_thrift_row_groups(pf, filename, rg_list)
        part = {'piece': (real_row_groups,)}
    else:
        full_path = fs.sep.join([p for p in [base_path, filename] if p != ''])
        row_groups = [rg[0] for rg in rg_list]
        part = {'piece': (full_path, row_groups)}
    return part