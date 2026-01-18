import dataclasses
import math
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple
import torch
@dataclasses.dataclass(frozen=True)
class GenerationState:
    token_ids: torch.Tensor
    kv_cache: torch.Tensor
    logits: torch.Tensor
    weights: torch.Tensor
    fsm_states: List[int]