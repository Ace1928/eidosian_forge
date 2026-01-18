import re
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import (
@staticmethod
def _normalize_general_and_western(sentence: str) -> str:
    """Apply a language-independent (general) tokenization."""
    sentence = f' {sentence} '
    rules = [('\\n-', ''), ('\\n', ' '), ('&quot;', '"'), ('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'), ('([{-~[-` -&(-+:-@/])', ' \\1 '), ("'s ", " 's "), ("'s$", " 's"), ('([^0-9])([\\.,])', '\\1 \\2 '), ('([\\.,])([^0-9])', ' \\1 \\2'), ('([0-9])(-)', '\\1 \\2 ')]
    for pattern, replacement in rules:
        sentence = re.sub(pattern, replacement, sentence)
    return sentence