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
def _get_input_to_perturb(input):
    if input.layout == torch._mkldnn:
        input_to_perturb = input.to_dense()
    elif _is_sparse_any_tensor(input):
        input_to_perturb = input.clone()
    else:
        input_to_perturb = input.data
    return input_to_perturb