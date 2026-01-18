import warnings
import cupy
from cupy_backends.cuda.api import runtime
import cupyx.scipy.signal._signaltools as filtering
from cupyx.scipy.signal._arraytools import (
from cupyx.scipy.signal.windows._windows import get_window
def _get_module_func_raw(module, func_name, *template_args):
    args_dtypes = [_get_raw_typename(arg.dtype) for arg in template_args]
    template = '_'.join(args_dtypes)
    kernel_name = f'{func_name}_{template}' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel