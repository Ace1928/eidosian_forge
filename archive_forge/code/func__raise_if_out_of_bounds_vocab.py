from typing import Tuple, Optional
from functools import cached_property
import torch
import torch.nn as nn
import torch.jit
def _raise_if_out_of_bounds_vocab(self, vocab_size: int, bonus_token_ids: torch.Tensor, draft_token_ids: torch.Tensor) -> None:
    assert torch.all(bonus_token_ids < vocab_size)
    assert torch.all(bonus_token_ids >= 0)
    assert torch.all(draft_token_ids < vocab_size)
    assert torch.all(draft_token_ids >= 0)