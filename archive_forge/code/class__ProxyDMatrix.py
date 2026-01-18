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
class _ProxyDMatrix(DMatrix):
    """A placeholder class when DMatrix cannot be constructed (QuantileDMatrix,
    inplace_predict).

    """

    def __init__(self) -> None:
        self.handle = ctypes.c_void_p()
        _check_call(_LIB.XGProxyDMatrixCreate(ctypes.byref(self.handle)))

    def _set_data_from_cuda_interface(self, data: DataType) -> None:
        """Set data from CUDA array interface."""
        interface = data.__cuda_array_interface__
        interface_str = bytes(json.dumps(interface), 'utf-8')
        _check_call(_LIB.XGProxyDMatrixSetDataCudaArrayInterface(self.handle, interface_str))

    def _set_data_from_cuda_columnar(self, data: DataType, cat_codes: list) -> None:
        """Set data from CUDA columnar format."""
        from .data import _cudf_array_interfaces
        interfaces_str = _cudf_array_interfaces(data, cat_codes)
        _check_call(_LIB.XGProxyDMatrixSetDataCudaColumnar(self.handle, interfaces_str))

    def _set_data_from_array(self, data: np.ndarray) -> None:
        """Set data from numpy array."""
        from .data import _array_interface
        _check_call(_LIB.XGProxyDMatrixSetDataDense(self.handle, _array_interface(data)))

    def _set_data_from_csr(self, csr: scipy.sparse.csr_matrix) -> None:
        """Set data from scipy csr"""
        from .data import _array_interface
        _LIB.XGProxyDMatrixSetDataCSR(self.handle, _array_interface(csr.indptr), _array_interface(csr.indices), _array_interface(csr.data), ctypes.c_size_t(csr.shape[1]))