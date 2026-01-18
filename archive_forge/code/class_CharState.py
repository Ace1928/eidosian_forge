import itertools
import os
import re
from string import Template
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
from tokenizers import Encoding, Tokenizer
class CharState:
    char_ix: Optional[int]

    def __init__(self, char_ix):
        self.char_ix = char_ix
        self.anno_ix: Optional[int] = None
        self.tokens: List[int] = []

    @property
    def token_ix(self):
        return self.tokens[0] if len(self.tokens) > 0 else None

    @property
    def is_multitoken(self):
        """
        BPE tokenizers can output more than one token for a char
        """
        return len(self.tokens) > 1

    def partition_key(self) -> CharStateKey:
        return CharStateKey(token_ix=self.token_ix, anno_ix=self.anno_ix)