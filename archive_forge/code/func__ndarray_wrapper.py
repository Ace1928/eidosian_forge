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
def _ndarray_wrapper(f, target_dtype, fp32_param=None, cond_arg=None):

    def _new_fun(*args, **kwargs):
        if cond_arg is not None:
            if cond_arg[0] not in kwargs or kwargs[cond_arg[0]] not in cond_arg[1]:
                return f(*args, **kwargs)
        if fp32_param:
            new_args = []
            for i, x in enumerate(args):
                if fp32_param[i]:
                    new_args.append(x)
                else:
                    new_args.append(_cast_symbol_NDArray(x, target_dtype))
        else:
            new_args = list(map(lambda x: _cast_symbol_NDArray(x, target_dtype), args))
        args = tuple(new_args)
        if fp32_param:
            new_kwargs = {}
            for k, v in kwargs.items():
                if k in fp32_param:
                    new_kwargs[k] = v
                else:
                    new_kwargs[k] = _cast_symbol_NDArray(v, target_dtype)
                kwargs = new_kwargs
        else:
            kwargs = {k: _cast_symbol_NDArray(v, target_dtype) for k, v in kwargs.items()}
        return f(*args, **kwargs)
    _new_fun.__name__ = f.__name__
    _new_fun.__module__ = f.__module__
    _new_fun.__doc__ = f.__doc__
    return _new_fun