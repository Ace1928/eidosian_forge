import json
import os
import random
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
import regex as re
from ....file_utils import ExplicitEnum, PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available
from ....tokenization_utils import AddedToken, PreTrainedTokenizer
from ....tokenization_utils_base import ENCODE_KWARGS_DOCSTRING, BatchEncoding, TextInput, TruncationStrategy
from ....utils import logging
def _tokenize(self, text):
    """Tokenize a string."""
    bpe_tokens = []
    for token in re.findall(self.pat, text):
        token = ''.join((self.byte_encoder[b] for b in token.encode('utf-8')))
        bpe_tokens.extend((bpe_token for bpe_token in self.bpe(token).split(' ')))
    return bpe_tokens