from typing import Callable, Dict, List, Optional, Tuple
import torch
from torchaudio.models import RNNT
from torchaudio.prototype.models.rnnt import TrieNode
def _get_hypo_predictor_out(hypo: Hypothesis) -> torch.Tensor:
    return hypo[1]