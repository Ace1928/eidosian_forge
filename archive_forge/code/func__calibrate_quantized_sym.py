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
def _calibrate_quantized_sym(qsym, th_dict):
    """Given a dictionary containing the thresholds for quantizing the layers,
    set the thresholds into the quantized symbol as the params of requantize operators.
    """
    if th_dict is None or len(th_dict) == 0:
        return qsym
    num_layer_outputs = len(th_dict)
    layer_output_names = []
    min_vals = []
    max_vals = []
    for k, v in th_dict.items():
        layer_output_names.append(k)
        min_vals.append(v[0])
        max_vals.append(v[1])
    calibrated_sym = SymbolHandle()
    check_call(_LIB.MXSetCalibTableToQuantizedSymbol(qsym.handle, mx_uint(num_layer_outputs), c_str_array(layer_output_names), c_array(ctypes.c_float, min_vals), c_array(ctypes.c_float, max_vals), ctypes.byref(calibrated_sym)))
    return Symbol(calibrated_sym)