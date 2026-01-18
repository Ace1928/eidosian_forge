from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.distributed.fsdp._shard_utils import (

        This is to be called before loading a *sharded* DTensor state dict.
        This gathers tensor in FSDP dimension and returns local tensor of
        TP DTensor.
        