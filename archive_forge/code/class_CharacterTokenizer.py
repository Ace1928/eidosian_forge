import collections
import copy
import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import is_sentencepiece_available, is_sudachi_projection_available, logging
class CharacterTokenizer:
    """Runs Character tokenization."""

    def __init__(self, vocab, unk_token, normalize_text=True):
        """
        Constructs a CharacterTokenizer.

        Args:
            **vocab**:
                Vocabulary object.
            **unk_token**: str
                A special symbol for out-of-vocabulary token.
            **normalize_text**: (`optional`) boolean (default True)
                Whether to apply unicode normalization to text before tokenization.
        """
        self.vocab = vocab
        self.unk_token = unk_token
        self.normalize_text = normalize_text

    def tokenize(self, text):
        """
        Tokenizes a piece of text into characters.

        For example, `input = "apple""` wil return as output `["a", "p", "p", "l", "e"]`.

        Args:
            text: A single token or whitespace separated tokens.
                This should have already been passed through *BasicTokenizer*.

        Returns:
            A list of characters.
        """
        if self.normalize_text:
            text = unicodedata.normalize('NFKC', text)
        output_tokens = []
        for char in text:
            if char not in self.vocab:
                output_tokens.append(self.unk_token)
                continue
            output_tokens.append(char)
        return output_tokens