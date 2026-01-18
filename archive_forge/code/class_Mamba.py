from typing import TYPE_CHECKING, Optional
import torch
from .transformers import TransformerTokenizer
class Mamba:
    """Represent a `mamba` model."""

    def __init__(self, model: 'MambaLMHeadModel', tokenizer: 'PreTrainedTokenizer', device):
        self.device = device
        self.model = model
        self.tokenizer = TransformerTokenizer(tokenizer)

    def forward(self, input_ids: torch.LongTensor, *_):
        """Compute a forward pass through the mamba model."""
        output = self.model(input_ids)
        next_token_logits = output.logits[..., -1, :]
        return (next_token_logits, None)

    def __call__(self, input_ids: torch.LongTensor, *_) -> torch.FloatTensor:
        return self.forward(input_ids)