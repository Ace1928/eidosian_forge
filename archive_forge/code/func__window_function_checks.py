from typing import Optional, Iterable
import torch
from math import sqrt
from torch import Tensor
from torch._torch_docs import factory_common_args, parse_kwargs, merge_dicts
def _window_function_checks(function_name: str, M: int, dtype: torch.dtype, layout: torch.layout) -> None:
    """Performs common checks for all the defined windows.
     This function should be called before computing any window.

     Args:
         function_name (str): name of the window function.
         M (int): length of the window.
         dtype (:class:`torch.dtype`): the desired data type of returned tensor.
         layout (:class:`torch.layout`): the desired layout of returned tensor.
     """
    if M < 0:
        raise ValueError(f'{function_name} requires non-negative window length, got M={M}')
    if layout is not torch.strided:
        raise ValueError(f'{function_name} is implemented for strided tensors only, got: {layout}')
    if dtype not in [torch.float32, torch.float64]:
        raise ValueError(f'{function_name} expects float32 or float64 dtypes, got: {dtype}')