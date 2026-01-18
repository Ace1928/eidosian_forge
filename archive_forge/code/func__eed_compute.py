import re
import unicodedata
from math import inf
from typing import List, Optional, Sequence, Tuple, Union
from torch import Tensor, stack, tensor
from typing_extensions import Literal
from torchmetrics.functional.text.helper import _validate_inputs
def _eed_compute(sentence_level_scores: List[Tensor]) -> Tensor:
    """Reduction for extended edit distance.

    Args:
        sentence_level_scores: list of sentence-level scores as floats

    Return:
        average of scores as a tensor

    """
    if len(sentence_level_scores) == 0:
        return tensor(0.0)
    return sum(sentence_level_scores) / tensor(len(sentence_level_scores))