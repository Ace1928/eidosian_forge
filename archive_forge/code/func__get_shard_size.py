import functools
import os
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
import torch
from torch import Tensor
import torch.distributed as dist
@functools.lru_cache()
def _get_shard_size(self, element_size: int, num_shards: int) -> int:
    if self.bucket_cap_mb <= 0:
        return 0
    MB = 1024 * 1024
    bucket_size = self.bucket_cap_mb * MB / element_size
    return int(bucket_size // num_shards)