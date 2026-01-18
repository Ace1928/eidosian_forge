import copy
import itertools
import math
from typing import Optional
import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d
from torch.distributed._shard.sharded_tensor import (
from torch.distributed._shard.sharding_spec import ShardMetadata
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard as DShard

    All gather a DTensor in its sharded dimension and return the local tensor.
    