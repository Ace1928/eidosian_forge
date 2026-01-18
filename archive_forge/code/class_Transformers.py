from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import torch
from outlines.models.tokenizer import Tokenizer
class Transformers:
    """Represents a `transformers` model."""

    def __init__(self, model: 'PreTrainedModel', tokenizer: 'PreTrainedTokenizer'):
        self.device = model.device
        self.model = model
        self.tokenizer = TransformerTokenizer(tokenizer)

    @torch.inference_mode
    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, past_key_values: Optional[Tuple]=None) -> Tuple[torch.FloatTensor, Optional[KVCacheType]]:
        """Compute a forward pass through the transformer model.

        Parameters
        ----------
        input_ids
            The input token ids.  Must be one or two dimensional.
        attention_mask
            The attention mask.  Must be one or two dimensional.
        past_key_values
            A tuple of tuples containing the cached key and value tensors for each
            attention head.

        Returns
        -------
        The computed logits and the new cached key and value tensors.

        """
        assert 0 < input_ids.ndim < 3
        if past_key_values:
            input_ids = input_ids[..., -1].unsqueeze(-1)
        output = self.model(input_ids, attention_mask=attention_mask, return_dict=True, output_attentions=False, output_hidden_states=False, past_key_values=past_key_values)
        return (output.logits, output.past_key_values)

    def __call__(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, past_key_values: Optional[Tuple]=None) -> torch.FloatTensor:
        logits, kv_cache = self.forward(input_ids, attention_mask, past_key_values)
        next_token_logits = logits[..., -1, :]
        return (next_token_logits, kv_cache)