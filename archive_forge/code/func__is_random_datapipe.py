import inspect
import warnings
from typing import Any, List, Optional, Set
import torch
from torch.utils.data.datapipes.iter.sharding import (
from torch.utils.data.graph import DataPipe, DataPipeGraph, traverse_dps
def _is_random_datapipe(datapipe: DataPipe) -> bool:
    if hasattr(datapipe, 'set_seed') and inspect.ismethod(datapipe.set_seed):
        return True
    return False