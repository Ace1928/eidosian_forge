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
def _compute_numerical_jvps_wrt_specific_input(jvp_fn, delta, input_is_complex, is_forward_ad=False) -> List[torch.Tensor]:
    jvps: List[torch.Tensor] = []
    ds_dx_tup = jvp_fn(delta[0] if isinstance(delta, tuple) else delta)
    if input_is_complex:
        ds_dy_tup = jvp_fn(delta[1] * 1j) if isinstance(delta, tuple) else jvp_fn(delta * 1j)
        for ds_dx, ds_dy in zip(ds_dx_tup, ds_dy_tup):
            assert not ds_dx.is_complex()
            conj_w_d = ds_dx + ds_dy * 1j
            jvps.append(conj_w_d)
    else:
        for ds_dx in ds_dx_tup:
            assert is_forward_ad or not ds_dx.is_complex()
            jvps.append(ds_dx)
    return jvps