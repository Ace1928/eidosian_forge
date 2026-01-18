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
def _split_tokens_on_spaces(tokenizer, tokens: List[int]):
    """Combine tokens into words by splitting at whitespace and punctuation tokens."""
    subwords, subword_tokens_list, subword_indices_list = _split_tokens_on_unicode(tokenizer, tokens)
    words = []
    word_tokens = []
    token_indices = []
    for subword, subword_tokens, subword_indices in zip(subwords, subword_tokens_list, subword_indices_list):
        special = subword_tokens[0] >= tokenizer.eos_token_id
        with_space = subword.startswith(' ')
        punctuation = subword.strip() in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        if special or with_space or punctuation or (len(words) == 0):
            words.append(subword)
            word_tokens.append(subword_tokens)
            token_indices.append(subword_indices)
        else:
            words[-1] = words[-1] + subword
            word_tokens[-1].extend(subword_tokens)
            token_indices[-1].extend(subword_indices)
    return (words, word_tokens, token_indices)