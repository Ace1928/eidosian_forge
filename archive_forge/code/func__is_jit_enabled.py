import torch
from ..modules import Module
from . import comm
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Sequence, Set, TypeVar, Union, cast
from torch._utils import _get_device_index
from collections import OrderedDict
def _is_jit_enabled() -> 'torch.jit._state.EnabledProxy':
    import torch.jit._state
    return torch.jit._state._enabled