import copy
import ctypes
import importlib.util
import json
import os
import re
import sys
import warnings
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import IntEnum, unique
from functools import wraps
from inspect import Parameter, signature
from typing import (
import numpy as np
import scipy.sparse
from ._typing import (
from .compat import PANDAS_INSTALLED, DataFrame, py_str
from .libpath import find_lib_path
def _get_feature_info(self, field: str) -> Optional[FeatureInfo]:
    length = c_bst_ulong()
    sarr = ctypes.POINTER(ctypes.c_char_p)()
    if not hasattr(self, 'handle') or self.handle is None:
        return None
    _check_call(_LIB.XGBoosterGetStrFeatureInfo(self.handle, c_str(field), ctypes.byref(length), ctypes.byref(sarr)))
    feature_info = from_cstr_to_pystr(sarr, length)
    return feature_info if feature_info else None