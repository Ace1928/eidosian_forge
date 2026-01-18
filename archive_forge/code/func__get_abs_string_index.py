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
def _get_abs_string_index(self, idx):
    """Get the absolute index for the list of modules."""
    idx = operator.index(idx)
    if not -len(self) <= idx < len(self):
        raise IndexError(f'index {idx} is out of range')
    if idx < 0:
        idx += len(self)
    return str(idx)