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
def _get_analytical_vjps_wrt_specific_output(vjp_fn, sample_output, v) -> List[List[Optional[torch.Tensor]]]:
    vjps: List[List[Optional[torch.Tensor]]] = []
    grad_inputs = vjp_fn(v.reshape(sample_output.shape))
    for vjp in grad_inputs:
        vjps.append([vjp.clone() if isinstance(vjp, torch.Tensor) else None])
    return vjps