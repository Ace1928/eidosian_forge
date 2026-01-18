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
def _quantize_params(qsym, params, th_dict):
    """Given a quantized symbol and a dict of params that have not been quantized,
    generate quantized params. Currently only supports quantizing the arg_params
    with names of `weight` or `bias`, not aux_params. If `qsym` contains symbols
    that are excluded from being quantized, their corresponding params will
    not be quantized, but saved together with quantized params of the symbols that
    have been quantized.

    Parameters
    ----------
    qsym : Symbol
        Quantized symbol from FP32 symbol.
    params : dict of str->NDArray
    th_dict: dict of min/max pairs of layers' output
    """
    inputs_name = qsym.list_arguments()
    quantized_params = {}
    for name in inputs_name:
        if name.endswith(('weight_quantize', 'bias_quantize')):
            original_name = name[:-len('_quantize')]
            param = params[original_name]
            val, vmin, vmax = ndarray.contrib.quantize(data=param, min_range=ndarray.min(param), max_range=ndarray.max(param), out_type='int8')
            quantized_params[name] = val
            quantized_params[name + '_min'] = vmin
            quantized_params[name + '_max'] = vmax
        elif name in params:
            quantized_params[name] = params[name]
        elif name.endswith('_min'):
            output = name[:-len('_min')]
            if output in th_dict:
                quantized_params[name] = ndarray.array([th_dict[output][0]])
        elif name.endswith('_max'):
            output = name[:-len('_min')]
            if output in th_dict:
                quantized_params[name] = ndarray.array([th_dict[output][1]])
    return quantized_params