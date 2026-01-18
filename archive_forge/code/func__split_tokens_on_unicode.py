import json
import os
import warnings
from functools import lru_cache
from typing import List, Optional, Tuple, Union
import numpy as np
import regex as re
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
from .english_normalizer import BasicTextNormalizer, EnglishTextNormalizer
def _split_tokens_on_unicode(tokenizer, tokens: List[int]):
    """Combine tokens into words by splitting at any position where the tokens are decoded as valid unicode points."""
    decoded_full = tokenizer.decode(tokens, decode_with_timestamps=True)
    replacement_char = '�'
    words = []
    word_tokens = []
    token_indices = []
    current_tokens = []
    current_indices = []
    unicode_offset = 0
    for token_idx, token in enumerate(tokens):
        current_tokens.append(token)
        current_indices.append(token_idx)
        decoded = tokenizer.decode(current_tokens, decode_with_timestamps=True)
        if replacement_char not in decoded or decoded_full[unicode_offset + decoded.index(replacement_char)] == replacement_char:
            words.append(decoded)
            word_tokens.append(current_tokens)
            token_indices.append(current_indices)
            current_tokens = []
            current_indices = []
            unicode_offset += len(decoded)
    return (words, word_tokens, token_indices)