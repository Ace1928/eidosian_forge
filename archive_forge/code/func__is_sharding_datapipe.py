import inspect
import warnings
from typing import Any, List, Optional, Set
import torch
from torch.utils.data.datapipes.iter.sharding import (
from torch.utils.data.graph import DataPipe, DataPipeGraph, traverse_dps
def _is_sharding_datapipe(datapipe: DataPipe) -> bool:
    if isinstance(datapipe, _ShardingIterDataPipe):
        return True
    if hasattr(datapipe, 'apply_sharding') and inspect.ismethod(datapipe.apply_sharding):
        return True
    return False