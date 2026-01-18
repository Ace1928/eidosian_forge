import re
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import (
def _handle_corner_cases_during_shifting(alignments: Dict[int, int], pred_errors: List[int], target_errors: List[int], pred_start: int, target_start: int, length: int) -> bool:
    """Return ``True`` if any of corner cases has been met. Otherwise, ``False`` is returned.

    Args:
        alignments: A dictionary mapping aligned positions between a reference and a hypothesis.
        pred_errors: A list of error positions in a hypothesis.
        target_errors: A list of error positions in a reference.
        pred_start: A hypothesis start index.
        target_start: A reference start index.
        length: A length of a word span to be considered.

    Return:
        An indication whether any of conrner cases has been met.

    """
    if sum(pred_errors[pred_start:pred_start + length]) == 0:
        return True
    if sum(target_errors[target_start:target_start + length]) == 0:
        return True
    if pred_start <= alignments[target_start] < pred_start + length:
        return True
    return False