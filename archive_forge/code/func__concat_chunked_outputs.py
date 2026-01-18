import torch
import functools
import threading
from torch import Tensor
from typing import Any, Callable, Optional, Tuple, Union, List
from torch.utils._pytree import (
from functools import partial
import os
import itertools
from torch._C._functorch import (
def _concat_chunked_outputs(out_dims, arg_spec, flat_output_chunks):
    flat_out_dims = _broadcast_to_and_flatten(out_dims, arg_spec)
    assert len(flat_out_dims) == len(flat_output_chunks)
    flat_output = []
    for idx, out_dim in enumerate(flat_out_dims):
        flat_output.append(torch.cat(flat_output_chunks[idx], dim=out_dim))
        flat_output_chunks[idx] = None
    return flat_output