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
def _cuda_array_interface(data: DataType) -> bytes:
    assert data.dtype.hasobject is False, 'Input data contains `object` dtype.  Expecting numeric data.'
    interface = data.__cuda_array_interface__
    if 'mask' in interface:
        interface['mask'] = interface['mask'].__cuda_array_interface__
    interface_str = bytes(json.dumps(interface), 'utf-8')
    return interface_str