import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as sp
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    if char == ' ' or char == '\t' or char == '\n' or (char == '\r'):
        return True
    cat = unicodedata.category(char)
    if cat == 'Zs':
        return True
    return False