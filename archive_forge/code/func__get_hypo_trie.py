from typing import Callable, Dict, List, Optional, Tuple
import torch
from torchaudio.models import RNNT
from torchaudio.prototype.models.rnnt import TrieNode
def _get_hypo_trie(hypo: Hypothesis) -> TrieNode:
    return hypo[4]