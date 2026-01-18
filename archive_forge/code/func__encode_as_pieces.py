import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
import sentencepiece as sp
from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging
def _encode_as_pieces(self, text):
    text = convert_to_unicode(text)
    if self.split_by_punct:
        words = self._run_split_on_punc(text)
        pieces = [self.spm.encode(w, out_type=str) for w in words]
        return [p for w in pieces for p in w]
    else:
        return self.spm.encode(text, out_type=str)