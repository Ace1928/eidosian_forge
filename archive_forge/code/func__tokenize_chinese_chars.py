import collections
import os
import unicodedata
from typing import List, Optional, Tuple
from ....tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ....utils import logging
def _tokenize_chinese_chars(self, text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        cp = ord(char)
        if self._is_chinese_char(cp):
            output.append(' ')
            output.append(char)
            output.append(' ')
        else:
            output.append(char)
    return ''.join(output)