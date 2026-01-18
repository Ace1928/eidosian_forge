from typing import Tuple, Optional
from functools import cached_property
import torch
import torch.nn as nn
import torch.jit
def _raise_if_incorrect_shape(self, target_probs: torch.Tensor, bonus_token_ids: torch.Tensor, draft_probs: torch.Tensor, draft_token_ids: torch.Tensor) -> None:
    target_batch_size, num_target_probs, target_vocab_size = target_probs.shape
    bonus_batch_size, num_bonus_tokens = bonus_token_ids.shape
    draft_batch_size, num_draft_probs, draft_vocab_size = draft_probs.shape
    draft_token_ids_batch_size, num_draft_token_ids = draft_token_ids.shape
    assert draft_batch_size == target_batch_size
    assert num_draft_probs == num_target_probs
    assert draft_vocab_size == target_vocab_size, f'draft_vocab_size={draft_vocab_size!r} target_vocab_size={target_vocab_size!r}'
    assert draft_token_ids_batch_size == draft_batch_size
    assert num_draft_token_ids == num_draft_probs
    assert bonus_batch_size == target_batch_size
    assert num_bonus_tokens == self._num_bonus_tokens