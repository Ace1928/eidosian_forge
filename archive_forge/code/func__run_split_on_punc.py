import collections
import os
import unicodedata
from typing import List, Optional, Tuple
from ....tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
from ....utils import logging
def _run_split_on_punc(self, text, never_split=None):
    """Splits punctuation on a piece of text."""
    if not self.do_split_on_punc or (never_split is not None and text in never_split):
        return [text]
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1
    return [''.join(x) for x in output]