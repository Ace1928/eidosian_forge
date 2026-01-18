import itertools
import warnings
from enum import auto, Enum
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp._common_utils import _FSDPState, _get_param_to_fqns
from torch.distributed.fsdp._flat_param import FlatParamHandle
def _get_names_from_handle_indices(self, handle_indices: Tuple[int, ...]) -> List[List[str]]:
    """
        Returns a list of FQNs for each handle in ``handle_indices``. If a
        handle index is invalid, then its FQNs are omitted from the returned
        list.
        """
    fqns: List[List[str]] = []
    for index in handle_indices:
        if index is None or index < 0 or index >= len(self.all_handles):
            continue
        handle = self.all_handles[index]
        flat_param = handle.flat_param
        fqns.append(self.param_to_fqn[flat_param])
    return fqns