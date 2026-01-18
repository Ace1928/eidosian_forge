from abc import ABC, abstractmethod
from dataclasses import dataclass
import functools
from typing import Callable, Dict, List, TYPE_CHECKING
import torch
from ._internals import (
from torch.distributed._shard.metadata import ShardMetadata
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from torch.distributed._shard.op_registry_utils import _decorator_func
def _has_custom_op(sharding_spec, op):
    """
    Returns whether or not the ShardingSpec has a custom op implementation.
    """
    class_name = type(sharding_spec).__qualname__
    return class_name in _CUSTOM_SHARDING_SPEC_OPS and op in _CUSTOM_SHARDING_SPEC_OPS[class_name]