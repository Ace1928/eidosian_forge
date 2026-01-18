import abc
import ctypes
import inspect
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from os import SEEK_END, environ
from os.path import getsize
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.sparse
from .compat import (PANDAS_INSTALLED, PYARROW_INSTALLED, arrow_cffi, arrow_is_floating, arrow_is_integer, concat,
from .libpath import find_lib_path
def _check_for_bad_pandas_dtypes(pandas_dtypes_series: pd_Series) -> None:
    bad_pandas_dtypes = [f'{column_name}: {pandas_dtype}' for column_name, pandas_dtype in pandas_dtypes_series.items() if not _is_allowed_numpy_dtype(pandas_dtype.type)]
    if bad_pandas_dtypes:
        raise ValueError(f'pandas dtypes must be int, float or bool.\nFields with bad pandas dtypes: {', '.join(bad_pandas_dtypes)}')