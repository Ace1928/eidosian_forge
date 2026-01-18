import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Type, TypeVar, Union
import torch
from torch.torch_version import TorchVersion
from typing_extensions import Annotated, get_args, get_origin
from .. import _is_triton_available
class BaseOperator:
    OPERATOR: Any
    NAME: str
    OPERATOR_CATEGORY: str

    @classmethod
    def is_available(cls) -> bool:
        if cls.OPERATOR is None or getattr(cls.OPERATOR, '__name__', '') == 'no_such_operator':
            return False
        return True

    @classmethod
    def operator_flop(cls, *inputs) -> int:
        """Calculate number of FLOP given inputs to `OPERATOR`"""
        return -1