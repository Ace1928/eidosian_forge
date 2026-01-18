from typing import Callable, Dict, List, Optional, Tuple
import torch
from torchaudio.models import RNNT
from torchaudio.prototype.models.rnnt import TrieNode
def _get_trie_mask(self, trie):
    step_mask = torch.ones(len(self.model.char_list) + 1)
    step_mask[list(trie[0].keys())] = 0
    return step_mask