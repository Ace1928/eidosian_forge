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
class ShardedTensorBase(torch.Tensor):
    _sharding_spec: shard_spec.ShardingSpec
    _metadata: ShardedTensorMetadata
    _local_shards: List[Shard]

    def __new__(cls, sharding_spec: shard_spec.ShardingSpec, *size, **kwargs):
        torch._C._log_api_usage_once('torch.distributed._shard.sharded_tensor')
        if not isinstance(sharding_spec, shard_spec.ShardingSpec):
            raise ValueError(f'Expecting ShardingSpec but got: {type(sharding_spec)}')
        sizes = _flatten_tensor_size(size)
        dtype = kwargs['dtype']
        layout = kwargs['layout']
        pin_memory = kwargs['pin_memory']
        requires_grad = kwargs['requires_grad']
        if dtype is None:
            dtype = torch.get_default_dtype()
        tensor_properties = TensorProperties(dtype, layout, requires_grad, pin_memory=pin_memory)
        sharded_tensor_metadata = sharding_spec.build_metadata(sizes, tensor_properties=tensor_properties)
        r = torch.Tensor._make_wrapper_subclass(cls, sizes, dtype=dtype, layout=layout, pin_memory=pin_memory, requires_grad=requires_grad)
        r._sharding_spec = sharding_spec
        r._metadata = sharded_tensor_metadata
        r._local_shards = []
        return r

    def metadata(self) -> ShardedTensorMetadata:
        """
        Returns a :class:`ShardedTensorMetadata` object corresponding to the
        metadata for the entire tensor.
        """
        return self._metadata

    def local_shards(self) -> List[Shard]:
        """
        Returns a list of :class:`Shard' corresponding to the
        local shards for this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return self._local_shards

    @classmethod
    def _init_from_local_shards_and_global_metadata(cls, local_shards: List[Shard], sharded_tensor_metadata: ShardedTensorMetadata, sharding_spec=None) -> ShardedTensorBase:
        """
        Initialize a ShardedTensorBase with local shards and a global
        ShardedTensorMetadata built on each rank.
        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties
        if len(shards_metadata) == 0:
            raise ValueError('shards_metadata must not be empty!')
        if tensor_properties.layout != torch.strided:
            raise ValueError('Only torch.strided layout is currently supported')
        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec
        sharded_tensor_base = ShardedTensorBase.__new__(ShardedTensor, spec, sharded_tensor_metadata.size, dtype=tensor_properties.dtype, layout=tensor_properties.layout, pin_memory=tensor_properties.pin_memory, requires_grad=tensor_properties.requires_grad)
        validate_non_overlapping_shards_metadata(shards_metadata)
        check_tensor(shards_metadata, list(sharded_tensor_metadata.size))
        sharded_tensor_base._local_shards = local_shards
        return sharded_tensor_base

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise RuntimeError(f'A {cls.__name__} object is being used from c++ while calling {func.__module__}.{func.__name__} but the there is no custom __torch_dispatch__ implementation for it.')