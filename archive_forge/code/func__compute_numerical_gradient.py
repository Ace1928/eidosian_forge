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
def _compute_numerical_gradient(fn, entry, v, norm_v, nbhd_checks_fn):
    if _is_sparse_compressed_tensor(entry):
        assert entry.layout == v.layout, (entry.layout, v.layout)
        assert entry._nnz() == v._nnz(), (entry._nnz(), v._nnz(), entry.shape)
        entry = entry.values()
        v = v.values()
        entry = entry.detach()
    orig = entry.clone()
    entry.copy_(orig - v)
    outa = fn()
    entry.copy_(orig + v)
    outb = fn()
    entry.copy_(orig)

    def compute(a, b):
        nbhd_checks_fn(a, b)
        ret = (b - a) / (2 * norm_v)
        return ret.detach().reshape(-1)
    return tuple((compute(a, b) for a, b in zip(outa, outb)))