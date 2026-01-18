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
class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):

    def __repr__(self):
        if not self.missing_keys and (not self.unexpected_keys):
            return '<All keys matched successfully>'
        return super().__repr__()
    __str__ = __repr__