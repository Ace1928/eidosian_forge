import re
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import (
def _shift_word_before_previous_position(words: List[str], start: int, target: int, length: int) -> List[str]:
    return words[:target] + words[start:start + length] + words[target:start] + words[start + length:]