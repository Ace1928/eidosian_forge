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
def _wrap_symbol_functions(module, target_dtype, target_precision_ops=None, conditional_fp32_ops=None, fp32_ops=None):

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

    def _symbol_wrapper(f, target_dtype, fp32_param=None, cond_arg=None):

        def _new_fun(*args, **kwargs):
            if cond_arg is not None:
                if cond_arg[0] not in kwargs or kwargs[cond_arg[0]] not in cond_arg[1]:
                    return f(*args, **kwargs)
            sym = f(*args, **kwargs)
            inputs = sym.get_children()
            aux = sym.list_auxiliary_states()
            if fp32_param:
                new_inputs = []
                for i, x in enumerate(inputs):
                    if x.name in aux or fp32_param[i]:
                        new_inputs.append(x)
                    else:
                        new_inputs.append(_cast_symbol_NDArray(x, target_dtype))
                inputs = new_inputs
            else:
                inputs = list(map(lambda x: _cast_symbol_NDArray(x, target_dtype) if x.name not in aux else x, inputs))
            atomic_sym = sym._gen_atomic_symbol()
            wrapped_sym = atomic_sym(*inputs)
            wrapped_sym._set_attr(name=sym.name)
            return wrapped_sym
        _new_fun.__name__ = f.__name__
        _new_fun.__module__ = f.__module__
        _new_fun.__doc__ = f.__doc__
        return _new_fun

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
    _wrapper = _symbol_wrapper if module in (symbol, Symbol, symbol_contrib) else _ndarray_wrapper
    submodule_dict = {}
    for op_name_prefix in base._OP_NAME_PREFIX_LIST:
        submodule_dict[op_name_prefix] = getattr(module, op_name_prefix[1:-1])
    fp32_param_list = list_lp16_use_fp32_params(target_dtype)
    wrap_list = target_precision_ops if target_precision_ops is not None else list_lp16_ops(target_dtype)
    for fun_name in wrap_list:
        try:
            fun_name, cur_module = _get_fun_to_wrap(fun_name, module, submodule_dict)
            f_to_wrap = getattr(cur_module, fun_name)
            fp32_param = fp32_param_list[fun_name] if fp32_param_list and fun_name in fp32_param_list else None
            setattr(cur_module, fun_name, _wrapper(f_to_wrap, target_dtype, fp32_param=fp32_param))
            if cur_module == module:
                setattr(module.op, fun_name, _wrapper(f_to_wrap, target_dtype, fp32_param=fp32_param))
        except AttributeError:
            raise
    wrap_list = fp32_ops if fp32_ops is not None else list_fp32_ops(target_dtype)
    for fun_name in wrap_list:
        try:
            fun_name, cur_module = _get_fun_to_wrap(fun_name, module, submodule_dict)
            f_to_wrap = getattr(cur_module, fun_name)
            setattr(cur_module, fun_name, _wrapper(f_to_wrap, np.float32))
            if cur_module == module:
                setattr(module.op, fun_name, _wrapper(f_to_wrap, np.float32))
        except AttributeError:
            raise
    wrap_list = conditional_fp32_ops if conditional_fp32_ops is not None else list_conditional_fp32_ops(target_dtype)
    for fun_name, arg, arg_values in wrap_list:
        try:
            fun_name, cur_module = _get_fun_to_wrap(fun_name, module, submodule_dict)
            f_to_wrap = getattr(cur_module, fun_name)
            setattr(cur_module, fun_name, _wrapper(f_to_wrap, np.float32, cond_arg=(arg, arg_values)))
            if cur_module == module:
                setattr(module.op, fun_name, _wrapper(f_to_wrap, np.float32, cond_arg=(arg, arg_values)))
        except AttributeError:
            raise
    for fun_name in list_widest_type_cast(target_dtype):
        try:
            fun_name, cur_module = _get_fun_to_wrap(fun_name, module, submodule_dict)
            f_to_wrap = getattr(cur_module, fun_name)
            setattr(cur_module, fun_name, _symbol_widest_wrapper(f_to_wrap))
            if cur_module == module:
                setattr(module.op, fun_name, _symbol_widest_wrapper(f_to_wrap))
        except AttributeError:
            raise