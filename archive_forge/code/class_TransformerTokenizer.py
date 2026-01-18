from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import torch
from outlines.models.tokenizer import Tokenizer
class TransformerTokenizer(Tokenizer):
    """Represents a tokenizer for models in the `transformers` library."""

    def __init__(self, tokenizer: 'PreTrainedTokenizer', **kwargs):
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.eos_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id
            self.pad_token = self.tokenizer.pad_token
        self.special_tokens = set(self.tokenizer.all_special_tokens)
        self.vocabulary = self.tokenizer.get_vocab()
        self.is_llama = isinstance(self.tokenizer, get_llama_tokenizer_types())

    def encode(self, prompt: Union[str, List[str]], **kwargs) -> Tuple[torch.LongTensor, torch.LongTensor]:
        kwargs['padding'] = True
        kwargs['return_tensors'] = 'pt'
        output = self.tokenizer(prompt, **kwargs)
        return (output['input_ids'], output['attention_mask'])

    def decode(self, token_ids: torch.LongTensor) -> List[str]:
        text = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return text

    def convert_token_to_string(self, token: str) -> str:
        from transformers.file_utils import SPIECE_UNDERLINE
        string = self.tokenizer.convert_tokens_to_string([token])
        if self.is_llama:
            if token.startswith(SPIECE_UNDERLINE) or token == '<0x20>':
                return ' ' + string
        return string

    def __eq__(self, other):
        if isinstance(other, type(self)):
            if hasattr(self, 'model_name') and hasattr(self, 'kwargs'):
                return other.model_name == self.model_name and other.kwargs == self.kwargs
            else:
                return other.tokenizer == self.tokenizer
        return NotImplemented

    def __hash__(self):
        from datasets.fingerprint import Hasher
        return hash(Hasher.hash(self.tokenizer))