import re
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import (
@staticmethod
def _remove_punct(sentence: str) -> str:
    """Remove punctuation from an input sentence string."""
    return re.sub('[\\.,\\?:;!\\"\\(\\)]', '', sentence)