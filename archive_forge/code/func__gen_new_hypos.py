from typing import Callable, Dict, List, Optional, Tuple
import torch
from torchaudio.models import RNNT
from torchaudio.prototype.models.rnnt import TrieNode
def _gen_new_hypos(self, base_hypos: List[Hypothesis], tokens: List[int], scores: List[float], t: int, device: torch.device) -> List[Hypothesis]:
    tgt_tokens = torch.tensor([[token] for token in tokens], device=device)
    states = _batch_state(base_hypos)
    pred_out, _, pred_states = self.model.predict(tgt_tokens, torch.tensor([1] * len(base_hypos), device=device), states)
    new_hypos: List[Hypothesis] = []
    for i, h_a in enumerate(base_hypos):
        new_tokens = _get_hypo_tokens(h_a) + [tokens[i]]
        if self.dobiasing:
            new_trie = self.model.get_tcpgen_step(tokens[i], _get_hypo_trie(h_a), self.resettrie)
        else:
            new_trie = self.resettrie
        new_hypos.append((new_tokens, pred_out[i].detach(), _slice_state(pred_states, i, device), scores[i], new_trie))
    return new_hypos