import re
import unicodedata
from math import inf
from typing import List, Optional, Sequence, Tuple, Union
from torch import Tensor, stack, tensor
from typing_extensions import Literal
from torchmetrics.functional.text.helper import _validate_inputs
def _distance_between_words(preds_word: str, target_word: str) -> int:
    """Distance measure used for substitutions/identity operation.

    Code adapted from https://github.com/rwth-i6/ExtendedEditDistance/blob/master/EED.py.

    Args:
        preds_word: hypothesis word string
        target_word: reference word string

    Return:
        0 for match, 1 for no match

    """
    return int(preds_word != target_word)