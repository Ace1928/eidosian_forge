import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
from safetensors import deserialize, safe_open, serialize, serialize_file
def _find_shared_tensors(state_dict: Dict[str, torch.Tensor]) -> List[Set[str]]:
    tensors = defaultdict(set)
    for k, v in state_dict.items():
        if v.device != torch.device('meta') and storage_ptr(v) != 0 and (storage_size(v) != 0):
            tensors[v.device, storage_ptr(v), storage_size(v)].add(k)
    tensors = list(sorted(tensors.values()))
    tensors = _filter_shared_not_shared(tensors, state_dict)
    return tensors