import threading
import copy
import warnings
import re
import json
from collections import OrderedDict, defaultdict
import numpy as np
from ..base import mx_real_t, MXNetError
from .. import symbol, ndarray, initializer, np_symbol
from ..symbol import Symbol, load_json
from ..ndarray import NDArray
from .. import name as _name
from .parameter import Parameter, ParameterDict, DeferredInitializationError
from .utils import _indent, _brief_print_list, HookHandle
from .utils import _check_same_symbol_type, _check_all_np_ndarrays
from .. import numpy_extension as _mx_npx
from .. import numpy as _mx_np
from .. util import is_np_array, np_shape, np_array
def _infer_param_types(in_params, out_params, arg_params, aux_params, default_dtype=mx_real_t):
    """Utility function that helps in inferring DType of args and auxs params
    from given input param.

    Parameters
    ----------
    in_params: List of Symbol
        List of input symbol variables.
    out_params: Symbol
        Output symbol variable.
    arg_params: List of Str
        List of names of argument parametrs.
    aux_params: List of Str
        List of names of auxiliary parameters.
    default_dtype: numpy.dtype or str, default 'float32'
        Default data type for arg_params and aux_params, if unable to infer the type.

    Returns
    -------
    arg_types: List of numpy.dtype
        List of arg_params type. Order is same as arg_params.
        Defaults to 'float32', if unable to infer type.
    aux_types: List of numpy.dtype
        List of aux_params type. Order is same as aux_params.
        Defaults to 'float32', if unable to infer type.
    """
    arg_types = None
    aux_types = None
    input_sym_names = [in_param.name for in_param in in_params]
    input_sym_arg_types = []
    can_infer_input_type = True
    for in_param in in_params:
        input_sym_arg_type = in_param.infer_type()[0]
        if not input_sym_arg_type or len(input_sym_arg_type) < 1:
            can_infer_input_type = False
            break
        else:
            input_sym_arg_types.append(in_param.infer_type()[0][0])
    if can_infer_input_type:
        params = {k: v for k, v in zip(input_sym_names, input_sym_arg_types)}
        try:
            arg_types, _, aux_types = out_params.infer_type(**params)
        except MXNetError:
            arg_types, aux_types = (None, None)
    if arg_types is None or len(arg_types) != len(arg_params):
        arg_types = []
        for _ in arg_params:
            arg_types.append(default_dtype)
    if aux_types is None or len(aux_types) != len(aux_params):
        aux_types = []
        for _ in aux_params:
            aux_types.append(default_dtype)
    return (arg_types, aux_types)