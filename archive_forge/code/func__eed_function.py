import re
import unicodedata
from math import inf
from typing import List, Optional, Sequence, Tuple, Union
from torch import Tensor, stack, tensor
from typing_extensions import Literal
from torchmetrics.functional.text.helper import _validate_inputs
def _eed_function(hyp: str, ref: str, alpha: float=2.0, rho: float=0.3, deletion: float=0.2, insertion: float=1.0) -> float:
    """Compute extended edit distance score for two lists of strings: hyp and ref.

    Code adapted from: https://github.com/rwth-i6/ExtendedEditDistance/blob/master/EED.py.

    Args:
        hyp: A hypothesis string
        ref: A reference string
        alpha: optimal jump penalty, penalty for jumps between characters
        rho: coverage cost, penalty for repetition of characters
        deletion: penalty for deletion of character
        insertion: penalty for insertion or substitution of character

    Return:
        Extended edit distance score as float
    """
    number_of_visits = [-1] * (len(hyp) + 1)
    row = [1.0] * (len(hyp) + 1)
    row[0] = 0.0
    next_row = [inf] * (len(hyp) + 1)
    for w in range(1, len(ref) + 1):
        for i in range(len(hyp) + 1):
            if i > 0:
                next_row[i] = min(next_row[i - 1] + deletion, row[i - 1] + _distance_between_words(hyp[i - 1], ref[w - 1]), row[i] + insertion)
            else:
                next_row[i] = row[i] + 1.0
        min_index = next_row.index(min(next_row))
        number_of_visits[min_index] += 1
        if ref[w - 1] == ' ':
            jump = alpha + next_row[min_index]
            next_row = [min(x, jump) for x in next_row]
        row = next_row
        next_row = [inf] * (len(hyp) + 1)
    coverage = rho * sum((x if x >= 0 else 1 for x in number_of_visits))
    return min(1, (row[-1] + coverage) / (float(len(ref)) + coverage))