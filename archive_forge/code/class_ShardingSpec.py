from abc import ABC, abstractmethod
from dataclasses import dataclass
import functools
from typing import Callable, Dict, List, TYPE_CHECKING
import torch
from ._internals import (
from torch.distributed._shard.metadata import ShardMetadata
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from torch.distributed._shard.op_registry_utils import _decorator_func
class ShardingSpec(ABC):
    """
    Base class representing sharding specifications.
    """

    @abstractmethod
    def build_metadata(self, tensor_sizes: torch.Size, tensor_properties: sharded_tensor_meta.TensorProperties) -> sharded_tensor_meta.ShardedTensorMetadata:
        """
        Given a global tensor size, define how to shard a tensor like this shape
        across ranks, return ShardedTensorMetadata
        Args:
            tensor_sizes (:class:`torch.Size`):
                The tensor shape to shard on, a `torch.Size` object that represents the
                tensor shape to be sharded according to the ShardingSpec.
            tensor_properties(:class:`torch.distributed._shard.sharded_tensor.TensorProperties):
                Tensor properties used to create a ShardedTensor.
        Returns:
            A :class:`ShardedTensorMetadata` object that encodes the information about
            the layout of the ShardedTensor and its properties.
        """

    @abstractmethod
    def shard(self, tensor: torch.Tensor, src_rank: int=0, process_group=None) -> 'ShardedTensor':
        """
        Given a global tensor on src_rank, shard this tensor
        across ranks within the process group, return a ShardedTensor.
        Args:
            tensor (:class:`torch.Tensor`): Tensor needs to be sharded.
        Keyword args:
            src_rank (int, optional): The source rank which is used as the ground truth of
                the data for the parameter that would be sharded and scattered
                across the rest of the ranks.
                Default: 0.
            process_group (ProcessGroup, optional): The process group to work on. If None,
                the default process group will be used.
        Returns:
            A :class:`ShardedTensor` sharded from the given tensor.
        """