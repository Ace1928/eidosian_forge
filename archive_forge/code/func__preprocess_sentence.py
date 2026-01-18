import re
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import (
def _preprocess_sentence(sentence: str, tokenizer: _TercomTokenizer) -> str:
    """Given a sentence, apply tokenization.

    Args:
        sentence: The input sentence string.
        tokenizer: An instance of ``_TercomTokenizer`` handling a sentence tokenization.

    Return:
        The pre-processed output sentence string.

    """
    return tokenizer(sentence.rstrip())