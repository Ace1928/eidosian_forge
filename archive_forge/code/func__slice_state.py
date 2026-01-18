from typing import Callable, Dict, List, Optional, Tuple
import torch
from torchaudio.models import RNNT
from torchaudio.prototype.models.rnnt import TrieNode
def _slice_state(states: List[List[torch.Tensor]], idx: int, device: torch.device) -> List[List[torch.Tensor]]:
    idx_tensor = torch.tensor([idx], device=device)
    return [[state.index_select(0, idx_tensor) for state in state_tuple] for state_tuple in states]