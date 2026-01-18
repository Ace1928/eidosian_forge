import ctypes
import logging
import os
import shutil
import warnings
import numpy as np
from ..base import _LIB, check_call, py_str
from ..base import c_array, c_str, mx_uint, c_str_array
from ..base import NDArrayHandle, SymbolHandle
from ..symbol import Symbol
from ..symbol import load as sym_load
from .. import ndarray
from ..ndarray import load as nd_load
from ..ndarray import save as nd_save
from ..ndarray import NDArray
from ..io import DataIter, DataDesc, DataBatch
from ..context import cpu, Context
from ..module import Module
def _as_data_iter(calib_data):
    """Convert normal iterator to mx.io.DataIter while parsing the data_shapes"""
    if isinstance(calib_data, DataIter):
        return (calib_data, calib_data.provide_data)
    calib_data = _DataIterWrapper(calib_data)
    return (calib_data, calib_data.provide_data)