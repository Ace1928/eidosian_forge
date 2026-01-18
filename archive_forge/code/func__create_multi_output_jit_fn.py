import re
from typing import Callable, List
import torch
from torch import Tensor
def _create_multi_output_jit_fn(code_string: str, num_outputs: int, **kwargs) -> Callable:
    """
    Create a jiterator-generated cuda kernel for an elementwise op that supports returning one or more outputs.

    Args:
        code_string (str): CUDA code string to be compiled by jiterator. The entry functor must return value by reference.
        num_outputs(int): number of outputs return by the kernel
        kwargs (Dict, optional): Keyword arguments for generated function

    Example::

        code_string = "template <typename T> void my_kernel(T x, T y, T alpha, T& out) { out = -x + alpha * y; }"
        jitted_fn = create_jit_fn(code_string, alpha=1.0)
        a = torch.rand(3, device='cuda')
        b = torch.rand(3, device='cuda')
        # invoke jitted function like a regular python function
        result = jitted_fn(a, b, alpha=3.14)

    .. warning::
        This API is in beta and may change in future releases.

    .. warning::
        This API only supports up to 8 inputs and 8 outputs
    """
    return _JittedFunction(code_string, return_by_ref=True, num_outputs=num_outputs, **kwargs)