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
def _dump_pandas_categorical(pandas_categorical: Optional[List[List]], file_name: Optional[Union[str, Path]]=None) -> str:
    categorical_json = json.dumps(pandas_categorical, default=_json_default_with_numpy)
    pandas_str = f'\npandas_categorical:{categorical_json}\n'
    if file_name is not None:
        with open(file_name, 'a') as f:
            f.write(pandas_str)
    return pandas_str