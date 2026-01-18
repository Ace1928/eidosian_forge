from collections import OrderedDict, namedtuple
import itertools
import warnings
import functools
import weakref
import torch
from torch._prims_common import DeviceLikeType
from ..parameter import Parameter
import torch.utils.hooks as hooks
from torch import Tensor, device, dtype
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from typing_extensions import Self
from ...utils.hooks import RemovableHandle
def _get_backward_hooks(self):
    """Return the backward hooks for use in the call function.

        It returns two lists, one with the full backward hooks and one with the non-full
        backward hooks.
        """
    full_backward_hooks: List[Callable] = []
    if _global_is_full_backward_hook is True:
        full_backward_hooks += _global_backward_hooks.values()
    if self._is_full_backward_hook is True:
        full_backward_hooks += self._backward_hooks.values()
    non_full_backward_hooks: List[Callable] = []
    if _global_is_full_backward_hook is False:
        non_full_backward_hooks += _global_backward_hooks.values()
    if self._is_full_backward_hook is False:
        non_full_backward_hooks += self._backward_hooks.values()
    return (full_backward_hooks, non_full_backward_hooks)