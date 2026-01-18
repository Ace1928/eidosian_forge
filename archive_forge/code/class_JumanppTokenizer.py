import collections
import copy
import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
from ...tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ...utils import is_sentencepiece_available, is_sudachi_projection_available, logging
class JumanppTokenizer:
    """Runs basic tokenization with jumanpp morphological parser."""

    def __init__(self, do_lower_case=False, never_split=None, normalize_text=True, trim_whitespace=False):
        """
        Constructs a JumanppTokenizer.

        Args:
            **do_lower_case**: (*optional*) boolean (default True)
                Whether to lowercase the input.
            **never_split**: (*optional*) list of str
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of tokens not to split.
            **normalize_text**: (*optional*) boolean (default True)
                Whether to apply unicode normalization to text before tokenization.
            **trim_whitespace**: (*optional*) boolean (default False)
                Whether to trim all whitespace, tab, newline from tokens.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split if never_split is not None else []
        self.normalize_text = normalize_text
        self.trim_whitespace = trim_whitespace
        try:
            import rhoknp
        except ImportError:
            raise ImportError('You need to install rhoknp to use JumanppTokenizer. See https://github.com/ku-nlp/rhoknp for installation.')
        self.juman = rhoknp.Jumanpp()

    def tokenize(self, text, never_split=None, **kwargs):
        """Tokenizes a piece of text."""
        if self.normalize_text:
            text = unicodedata.normalize('NFKC', text)
        text = text.strip()
        never_split = self.never_split + (never_split if never_split is not None else [])
        tokens = []
        for mrph in self.juman.apply_to_sentence(text).morphemes:
            token = mrph.text
            if self.do_lower_case and token not in never_split:
                token = token.lower()
            if self.trim_whitespace:
                if token.strip() == '':
                    continue
                else:
                    token = token.strip()
            tokens.append(token)
        return tokens