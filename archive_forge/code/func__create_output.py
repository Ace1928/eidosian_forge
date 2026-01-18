from typing import Tuple, Optional
from functools import cached_property
import torch
import torch.nn as nn
import torch.jit
def _create_output(self, accepted: torch.Tensor, recovered_token_ids: torch.Tensor, draft_token_ids: torch.Tensor, bonus_token_ids: torch.Tensor) -> torch.Tensor:
    """Format output. Returns a matrix of token ids. When
        a token is rejected via rejection sampling, all subsequent
        token ids are set to -1 for the sequence.

        shape = [batch_size, k + num_bonus_tokens]
        """
    bonus_token_ids = bonus_token_ids.squeeze()
    batch_size, k = recovered_token_ids.shape
    limits = (accepted == 0).max(1).indices
    limits[~(accepted == 0).any(1)] = k
    indices = torch.arange(k, device=accepted.device).unsqueeze(0)
    accepted_mask = indices < limits.unsqueeze(1)
    after_false_mask = indices == limits.unsqueeze(1)
    output_with_bonus_tokens = -torch.ones((batch_size, k + self._num_bonus_tokens), dtype=self.token_id_dtype, device=accepted.device)
    output = output_with_bonus_tokens[:, :k]
    output[:, :k] = torch.where(accepted_mask, draft_token_ids, -torch.ones_like(draft_token_ids))
    output_with_bonus_tokens[:, -1] = torch.where(output[:, -1] != -1, bonus_token_ids, -1)
    output.mul_(~after_false_mask).add_(recovered_token_ids.mul(after_false_mask))
    self.num_accepted_tokens += accepted.sum()
    self.num_emitted_tokens += (output_with_bonus_tokens != -1).sum()
    self.num_draft_tokens += batch_size * k
    return output_with_bonus_tokens