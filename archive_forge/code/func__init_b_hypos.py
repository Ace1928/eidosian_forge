from typing import Callable, Dict, List, Optional, Tuple
import torch
from torchaudio.models import RNNT
from torchaudio.prototype.models.rnnt import TrieNode
def _init_b_hypos(self, hypo: Optional[Hypothesis], device: torch.device) -> List[Hypothesis]:
    if hypo is not None:
        token = _get_hypo_tokens(hypo)[-1]
        state = _get_hypo_state(hypo)
    else:
        token = self.blank
        state = None
    one_tensor = torch.tensor([1], device=device)
    pred_out, _, pred_state = self.model.predict(torch.tensor([[token]], device=device), one_tensor, state)
    init_hypo = ([token], pred_out[0].detach(), pred_state, 0.0, self.resettrie)
    return [init_hypo]