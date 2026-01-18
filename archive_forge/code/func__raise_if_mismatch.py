from __future__ import annotations  # type: ignore[attr-defined]
from dataclasses import dataclass
from typing import (
import copy
import warnings
from functools import reduce
import weakref
import threading
import torch
import torch.distributed as dist
from torch.distributed import rpc
from torch.distributed import distributed_c10d
from torch.distributed._shard.metadata import ShardMetadata
import torch.distributed._shard.sharding_spec as shard_spec
from torch.distributed._shard.sharding_spec.api import (
from torch.distributed._shard.sharding_spec._internals import (
from torch.distributed._shard._utils import (
from .metadata import TensorProperties, ShardedTensorMetadata
from .shard import Shard
from .reshard import reshuffle_local_shard, reshard_local_shard
from .utils import (
from torch.distributed.remote_device import _remote_device
from torch.utils import _pytree as pytree
def _raise_if_mismatch(expected, actual, prop_name, rank, is_property=False):
    tensor_property_or_metadata = 'tensor property' if is_property else 'local ShardMetadata'
    if expected != actual:
        raise ValueError(f"Local shards' tensor {prop_name} property is incompatible with {tensor_property_or_metadata} on rank {rank}: {tensor_property_or_metadata} {prop_name}={expected}, local shard tensor {prop_name}={actual}.")