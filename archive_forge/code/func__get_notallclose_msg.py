import collections
import functools
import warnings
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.testing
from torch._vmap_internals import _vmap, vmap
from torch.overrides import is_tensor_like
from torch.types import _TensorOrTensors
def _get_notallclose_msg(analytical, numerical, output_idx, input_idx, complex_indices, test_imag=False, is_forward_ad=False) -> str:
    out_is_complex = not is_forward_ad and complex_indices and (output_idx in complex_indices)
    inp_is_complex = is_forward_ad and complex_indices and (input_idx in complex_indices)
    part = 'imaginary' if test_imag else 'real'
    element = 'inputs' if is_forward_ad else 'outputs'
    prefix = '' if not (out_is_complex or inp_is_complex) else f'While considering the {part} part of complex {element} only, '
    mode = 'computed with forward mode ' if is_forward_ad else ''
    return prefix + 'Jacobian %smismatch for output %d with respect to input %d,\nnumerical:%s\nanalytical:%s\n' % (mode, output_idx, input_idx, numerical, analytical)