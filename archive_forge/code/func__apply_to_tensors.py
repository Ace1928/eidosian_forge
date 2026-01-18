import dataclasses
import traceback
from typing import Any, Callable, Container, Dict, List, Optional, OrderedDict, Tuple, TypeVar, overload
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel._functions import _get_stream
from torch.nn.parallel.scatter_gather import _is_namedtuple
from torch.nn.utils.rnn import PackedSequence
def _apply_to_tensors(fn, container):
    """Recursively apply to all tensor in different kinds of container types."""

    def apply(x):
        if isinstance(x, torch.Tensor):
            return fn(x)
        elif hasattr(x, '__dataclass_fields__'):
            dc = dataclasses.replace(x)
            for f in dataclasses.fields(dc):
                name = f.name
                setattr(dc, name, apply(getattr(dc, name)))
            return dc
        elif isinstance(x, OrderedDict):
            od = x.__class__()
            for key, value in x.items():
                od[key] = apply(value)
            return od
        elif isinstance(x, PackedSequence):
            apply(x.data)
            return x
        elif isinstance(x, dict):
            return {key: apply(value) for key, value in x.items()}
        elif _is_namedtuple(x):
            res = (apply(el) for el in x)
            return type(x)(*res)
        elif isinstance(x, (list, tuple, set)):
            return type(x)((apply(el) for el in x))
        else:
            return x
    return apply(container)