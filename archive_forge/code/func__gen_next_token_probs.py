from typing import Callable, Dict, List, Optional, Tuple
import torch
from torchaudio.models import RNNT
from torchaudio.prototype.models.rnnt import TrieNode
def _gen_next_token_probs(self, enc_out: torch.Tensor, hypos: List[Hypothesis], device: torch.device) -> torch.Tensor:
    one_tensor = torch.tensor([1], device=device)
    predictor_out = torch.stack([_get_hypo_predictor_out(h) for h in hypos], dim=0)
    if self.dobiasing:
        trie_masks = torch.stack([self._get_trie_mask(_get_hypo_trie(h)) for h in hypos], dim=0)
        trie_masks = trie_masks.to(enc_out.device).unsqueeze(1)
        genprob_masks = torch.tensor([self._get_generation_prob(_get_hypo_trie(h)) for h in hypos])
        genprob_masks = genprob_masks.to(enc_out.device)
        last_tokens = torch.tensor([_get_hypo_tokens(h)[-1] for h in hypos]).unsqueeze(-1).to(enc_out.device)
        hptr, tcpgen_dist = self.model.forward_tcpgen(last_tokens, trie_masks, enc_out)
    else:
        hptr = None
    joined_out, _, joined_activation = self.model.join(enc_out, one_tensor, predictor_out, torch.tensor([1] * len(hypos), device=device), hptr=hptr)
    if self.dobiasing:
        p_gen = torch.sigmoid(self.model.pointer_gate(torch.cat((joined_activation, hptr), dim=-1)))
        p_gen = p_gen.masked_fill(genprob_masks.view(p_gen.size(0), 1, 1, 1), 0)
        model_tu = torch.softmax(joined_out / self.temperature, dim=3)
        p_not_null = 1.0 - model_tu[:, :, :, -1:]
        ptr_dist_fact = torch.cat([tcpgen_dist[:, :, :, :-2], tcpgen_dist[:, :, :, -1:]], dim=-1) * p_not_null
        ptr_gen_complement = tcpgen_dist[:, :, :, -1:] * p_gen
        p_partial = ptr_dist_fact[:, :, :, :-1] * p_gen + model_tu[:, :, :, :-1] * (1 - p_gen + ptr_gen_complement)
        p_final = torch.cat([p_partial, model_tu[:, :, :, -1:]], dim=-1)
        joined_out = torch.log(p_final)
    else:
        joined_out = torch.nn.functional.log_softmax(joined_out / self.temperature, dim=3)
    return joined_out[:, 0, 0]