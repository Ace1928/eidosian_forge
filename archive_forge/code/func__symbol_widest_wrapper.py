from array import array
import ctypes
import logging
import contextlib
import numpy as np
from ... import symbol
from ...context import gpu
from ...symbol import Symbol
from ...module import BucketingModule
from ...symbol import contrib as symbol_contrib
from ... import ndarray
from ...ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from . import lists
from ...gluon import trainer
from ... import base
from ...base import c_str_array, SymbolHandle, check_call, _LIB, mx_uint, c_array_buf
from ... import optimizer as opt
from .loss_scaler import LossScaler
def _symbol_widest_wrapper(f):

    def _new_fun(*args, **kwargs):
        symbols = []
        is_symbol = False
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, (Symbol, NDArray)):
                symbols.append((args, i, arg))
                is_symbol = is_symbol or isinstance(arg, Symbol)
        for k, arg in kwargs.items():
            if isinstance(arg, (Symbol, NDArray)):
                symbols.append((kwargs, k, arg))
                is_symbol = is_symbol or isinstance(arg, Symbol)
        if not is_symbol:
            widest_type = target_dtype
            for _, _, arg in symbols:
                if isinstance(arg, NDArray):
                    if arg.dtype == np.float32:
                        widest_type = np.float32
            for arr, index, arg in symbols:
                if arg.dtype != widest_type and arg.dtype == target_dtype:
                    arr[index] = ndarray.amp_cast(arg, dtype=widest_type)
        else:
            sym_to_check = list(map(lambda x: x[2], symbols))
            casted_syms = symbol.amp_multicast(*sym_to_check, num_outputs=len(sym_to_check))
            symbols = list(map(lambda x_y: (x_y[0][0], x_y[0][1], x_y[1]), zip(symbols, casted_syms)))
            for arr, index, arg in symbols:
                arr[index] = arg
        return f(*args, **kwargs)
    _new_fun.__name__ = f.__name__
    _new_fun.__module__ = f.__module__
    _new_fun.__doc__ = f.__doc__
    return _new_fun