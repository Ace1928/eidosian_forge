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
def _get_chunked_inputs(flat_args, flat_in_dims, batch_size, chunk_size):
    split_idxs = (batch_size,)
    if chunk_size is not None:
        chunk_sizes = get_chunk_sizes(batch_size, chunk_size)
        split_idxs = tuple(itertools.accumulate(chunk_sizes))
    flat_args_chunks = tuple((t.tensor_split(split_idxs, dim=in_dim) if in_dim is not None else [t] * len(split_idxs) for t, in_dim in zip(flat_args, flat_in_dims)))
    chunks_flat_args = zip(*flat_args_chunks)
    return chunks_flat_args