import dataclasses
import traceback
from typing import Any, Callable, Container, Dict, List, Optional, OrderedDict, Tuple, TypeVar, overload
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel._functions import _get_stream
from torch.nn.parallel.scatter_gather import _is_namedtuple
from torch.nn.utils.rnn import PackedSequence
def _unpack_kwargs(flat_args: Tuple[Any, ...], kwarg_keys: Tuple[str, ...]) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """See _pack_kwargs."""
    assert len(kwarg_keys) <= len(flat_args), f'too many keys {len(kwarg_keys)} vs. {len(flat_args)}'
    if len(kwarg_keys) == 0:
        return (flat_args, {})
    args = flat_args[:-len(kwarg_keys)]
    kwargs = dict(zip(kwarg_keys, flat_args[-len(kwarg_keys):]))
    return (args, kwargs)