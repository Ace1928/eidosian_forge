import dataclasses
import traceback
from typing import Any, Callable, Container, Dict, List, Optional, OrderedDict, Tuple, TypeVar, overload
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel._functions import _get_stream
from torch.nn.parallel.scatter_gather import _is_namedtuple
from torch.nn.utils.rnn import PackedSequence
def _replace_by_prefix(state_dict: Dict[str, Any], old_prefix: str, new_prefix: str) -> None:
    """
    Replace all keys that match a given old_prefix with a new_prefix (in-place).

    Usage::

        state_dict = {"layer.xyz": torch.tensor(1)}
        replace_by_prefix_(state_dict, "layer.", "module.layer.")
        assert state_dict == {"module.layer.xyz": torch.tensor(1)}
    """
    if old_prefix == new_prefix:
        raise ValueError('old_prefix and new_prefix must be distinct')
    for key in list(state_dict.keys()):
        if not key.startswith(old_prefix):
            continue
        new_key = new_prefix + key[len(old_prefix):]
        state_dict[new_key] = state_dict[key]
        del state_dict[key]