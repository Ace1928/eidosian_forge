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
def _real_and_imag_output(fn):

    def apply_to_c_outs(fn, fn_to_apply):

        def wrapped_fn(*inputs):
            outs = _as_tuple(fn(*inputs))
            return tuple((fn_to_apply(o) if o.is_complex() else o for o in outs))
        return wrapped_fn
    return (apply_to_c_outs(fn, torch.real), apply_to_c_outs(fn, torch.imag))