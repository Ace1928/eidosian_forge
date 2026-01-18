import dataclasses
import traceback
from typing import Any, Callable, Container, Dict, List, Optional, OrderedDict, Tuple, TypeVar, overload
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel._functions import _get_stream
from torch.nn.parallel.scatter_gather import _is_namedtuple
from torch.nn.utils.rnn import PackedSequence
def _cast_forward_inputs(dtype: Optional[torch.dtype], *args: Any, **kwargs: Any) -> Tuple[Any, Any]:
    """
    Cast floating point tensors in ``args`` and ``kwargs`` to ``input_dtype``.

    This respects the existing ``requires_grad`` on the tensors.
    """
    if dtype is None:
        return (args, kwargs)

    def cast_fn(x: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(x) or x.dtype == dtype:
            return x
        return x.to(dtype)
    return (_apply_to_tensors(cast_fn, args), _apply_to_tensors(cast_fn, kwargs))