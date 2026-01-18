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
class DeviceQuantileDMatrix(QuantileDMatrix):
    """Use `QuantileDMatrix` instead.

    .. deprecated:: 1.7.0

    .. versionadded:: 1.1.0

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn('Please use `QuantileDMatrix` instead.', FutureWarning)
        super().__init__(*args, **kwargs)