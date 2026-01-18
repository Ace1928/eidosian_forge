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
def _get_split_feature(tree: Dict[str, Any], feature_names: Optional[List[str]]) -> Optional[str]:
    if _is_split_node(tree):
        if feature_names is not None:
            feature_name = feature_names[tree['split_feature']]
        else:
            feature_name = tree['split_feature']
    else:
        feature_name = None
    return feature_name