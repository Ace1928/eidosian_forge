import collections
from itertools import repeat
from typing import List, Dict, Any
def _list_with_default(out_size: List[int], defaults: List[int]) -> List[int]:
    import torch
    if isinstance(out_size, (int, torch.SymInt)):
        return out_size
    if len(defaults) <= len(out_size):
        raise ValueError(f'Input dimension should be at least {len(out_size) + 1}')
    return [v if v is not None else d for v, d in zip(out_size, defaults[-len(out_size):])]