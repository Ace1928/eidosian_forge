import operator
import cupy
from cupy._core import internal
from cupy._core._scalar import get_typename
from cupyx.scipy.sparse import csr_matrix
import numpy as np
def _get_module_func(module, func_name, *template_args):

    def _get_typename(dtype):
        typename = get_typename(dtype)
        if dtype.kind == 'c':
            typename = 'thrust::' + typename
        return typename
    args_dtypes = [_get_typename(arg.dtype) for arg in template_args]
    template = ', '.join(args_dtypes)
    kernel_name = f'{func_name}<{template}>' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel