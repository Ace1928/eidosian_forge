from typing import Callable, Dict, List, Optional, Tuple
import torch
from torchaudio.models import RNNT
from torchaudio.prototype.models.rnnt import TrieNode
def _default_hypo_sort_key(hypo: Hypothesis) -> float:
    return _get_hypo_score(hypo) / (len(_get_hypo_tokens(hypo)) + 1)