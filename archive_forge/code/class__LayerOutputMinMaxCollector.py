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
class _LayerOutputMinMaxCollector(object):
    """Saves layer output min and max values in a dict with layer names as keys.
    The collected min and max values will be directly used as thresholds for quantization.
    """

    def __init__(self, quantized_dtype, include_layer=None, logger=None):
        self.min_max_dict = {}
        self.quantized_dtype = quantized_dtype
        self.include_layer = include_layer
        self.logger = logger

    def collect(self, name, arr):
        """Callback function for collecting min and max values from an NDArray."""
        name = py_str(name)
        if name not in self.include_layer:
            return
        handle = ctypes.cast(arr, NDArrayHandle)
        arr = NDArray(handle, writable=False)
        min_range = ndarray.min(arr).asscalar()
        max_range = ndarray.max(arr).asscalar()
        if name in self.min_max_dict:
            cur_min_max = self.min_max_dict[name]
            self.min_max_dict[name] = (min(cur_min_max[0], min_range), max(cur_min_max[1], max_range))
        else:
            self.min_max_dict[name] = (min_range, max_range)
        if self.logger:
            self.logger.debug('Collecting layer %s min_range=%f, max_range=%f' % (name, min_range, max_range))