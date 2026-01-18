import warnings
from collections import OrderedDict, abc as container_abcs
from itertools import chain, islice
import operator
import torch
from .module import Module
from ..parameter import Parameter
from torch._jit_internal import _copy_to_script_wrapper
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, overload, Tuple, TypeVar, Union
from typing_extensions import Self
def _get_item_by_idx(self, iterator, idx) -> T:
    """Get the idx-th item of the iterator."""
    size = len(self)
    idx = operator.index(idx)
    if not -size <= idx < size:
        raise IndexError(f'index {idx} is out of range')
    idx %= size
    return next(islice(iterator, idx, None))