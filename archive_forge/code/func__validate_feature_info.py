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
def _validate_feature_info(feature_info: Sequence[str], n_features: int, name: str) -> List[str]:
    if isinstance(feature_info, str) or not isinstance(feature_info, Sequence):
        raise TypeError(f'Expecting a sequence of strings for {name}, got: {type(feature_info)}')
    feature_info = list(feature_info)
    if len(feature_info) != n_features and n_features != 0:
        msg = (f'{name} must have the same length as the number of data columns, ', f'expected {n_features}, got {len(feature_info)}')
        raise ValueError(msg)
    return feature_info