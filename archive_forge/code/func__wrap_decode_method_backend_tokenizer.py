from typing import List, Optional, Tuple
from tokenizers import pre_tokenizers
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_clip import CLIPTokenizer
def _wrap_decode_method_backend_tokenizer(self):
    orig_decode_method = self.backend_tokenizer.decode

    def new_decode_method(*args, **kwargs):
        text = orig_decode_method(*args, **kwargs)
        text = text.replace(self.backend_tokenizer.model.end_of_word_suffix, ' ').strip()
        return text
    self.backend_tokenizer.decode = new_decode_method