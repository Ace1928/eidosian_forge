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
def _set_data_from_csr(self, csr: scipy.sparse.csr_matrix) -> None:
    """Set data from scipy csr"""
    from .data import _array_interface
    _LIB.XGProxyDMatrixSetDataCSR(self.handle, _array_interface(csr.indptr), _array_interface(csr.indices), _array_interface(csr.data), ctypes.c_size_t(csr.shape[1]))