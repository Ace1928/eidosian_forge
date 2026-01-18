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
def _get_optimal_thresholds(hist_dict, quantized_dtype, num_quantized_bins=255, logger=None):
    """Given a ndarray dict, find the optimal threshold for quantizing each value of the key."""
    if stats is None:
        raise ImportError('scipy.stats is required for running entropy mode of calculating the optimal thresholds for quantizing FP32 ndarrays into int8. Please check if the scipy python bindings are installed.')
    assert isinstance(hist_dict, dict)
    if logger is not None:
        logger.info('Calculating optimal thresholds for quantization using KL divergence with num_quantized_bins=%d' % num_quantized_bins)
    th_dict = {}
    layer_names = list(hist_dict.keys())
    for name in layer_names:
        assert name in hist_dict
        min_val, max_val, th, divergence = _get_optimal_threshold(hist_dict[name], quantized_dtype, num_quantized_bins=num_quantized_bins)
        if min_val >= 0 and quantized_dtype in ['auto', 'uint8']:
            th_dict[name] = (0, th)
        else:
            th_dict[name] = (-th, th)
        del hist_dict[name]
        if logger:
            logger.debug('layer=%s, min_val=%f, max_val=%f, th=%f, divergence=%f' % (name, min_val, max_val, th, divergence))
    return th_dict