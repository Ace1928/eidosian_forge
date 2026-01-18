import os
import io
import itertools
from typing import (
import torch.distributed as dist
from .api import (
import torch
from torch.distributed._shard.sharded_tensor import (
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._tensor import DTensor
from .metadata import (
def _get_failure_dict(results: List[Union[T, WRAPPED_EXCEPTION]]) -> Dict[int, WRAPPED_EXCEPTION]:
    return cast(Dict[int, WRAPPED_EXCEPTION], {i: err for i, err in enumerate(results) if _is_wrapped_exception(err)})