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
def _set_feature_info(self, features: Optional[FeatureInfo], field: str) -> None:
    if features is not None:
        assert isinstance(features, list)
        feature_info_bytes = [bytes(f, encoding='utf-8') for f in features]
        c_feature_info = (ctypes.c_char_p * len(feature_info_bytes))(*feature_info_bytes)
        _check_call(_LIB.XGBoosterSetStrFeatureInfo(self.handle, c_str(field), c_feature_info, c_bst_ulong(len(features))))
    else:
        _check_call(_LIB.XGBoosterSetStrFeatureInfo(self.handle, c_str(field), None, c_bst_ulong(0)))