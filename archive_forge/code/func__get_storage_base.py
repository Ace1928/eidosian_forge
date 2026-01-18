import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Type, TypeVar, Union
import torch
from torch.torch_version import TorchVersion
from typing_extensions import Annotated, get_args, get_origin
from .. import _is_triton_available
def _get_storage_base(x: torch.Tensor) -> int:
    return _GET_TENSOR_STORAGE(x).data_ptr()