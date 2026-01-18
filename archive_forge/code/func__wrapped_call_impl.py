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
def _wrapped_call_impl(self, *args, **kwargs):
    if self._compiled_call_impl is not None:
        return self._compiled_call_impl(*args, **kwargs)
    else:
        return self._call_impl(*args, **kwargs)