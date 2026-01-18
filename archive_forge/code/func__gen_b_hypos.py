from typing import Callable, Dict, List, Optional, Tuple
import torch
from torchaudio.models import RNNT
from torchaudio.prototype.models.rnnt import TrieNode
def _gen_b_hypos(self, b_hypos: List[Hypothesis], a_hypos: List[Hypothesis], next_token_probs: torch.Tensor, key_to_b_hypo: Dict[str, Hypothesis]) -> List[Hypothesis]:
    for i in range(len(a_hypos)):
        h_a = a_hypos[i]
        append_blank_score = _get_hypo_score(h_a) + next_token_probs[i, -1]
        if _get_hypo_key(h_a) in key_to_b_hypo:
            h_b = key_to_b_hypo[_get_hypo_key(h_a)]
            _remove_hypo(h_b, b_hypos)
            score = float(torch.tensor(_get_hypo_score(h_b)).logaddexp(append_blank_score))
        else:
            score = float(append_blank_score)
        h_b = (_get_hypo_tokens(h_a), _get_hypo_predictor_out(h_a), _get_hypo_state(h_a), score, _get_hypo_trie(h_a))
        b_hypos.append(h_b)
        key_to_b_hypo[_get_hypo_key(h_b)] = h_b
    _, sorted_idx = torch.tensor([_get_hypo_score(hypo) for hypo in b_hypos]).sort()
    return [b_hypos[idx] for idx in sorted_idx]