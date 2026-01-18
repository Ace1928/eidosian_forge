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
def _key_to_attr(self, key: str) -> str:
    if not isinstance(key, str):
        raise TypeError(f"Index given to ParameterDict cannot be used as a key as it is not a string (type is '{type(key).__name__}'). Open an issue on github if you need non-string keys.")
    else:
        return key