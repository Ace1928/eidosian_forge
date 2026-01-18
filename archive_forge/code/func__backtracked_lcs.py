import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.imports import _NLTK_AVAILABLE
def _backtracked_lcs(lcs_table: Sequence[Sequence[int]], pred_tokens: Sequence[str], target_tokens: Sequence[str]) -> Sequence[int]:
    """Backtrack LCS table.

    Args:
        lcs_table: A table containing information for the calculation of the longest common subsequence.
        pred_tokens: A tokenized predicted sentence.
        target_tokens: A tokenized target sentence.

    """
    i = len(pred_tokens)
    j = len(target_tokens)
    backtracked_lcs: List[int] = []
    while i > 0 and j > 0:
        if pred_tokens[i - 1] == target_tokens[j - 1]:
            backtracked_lcs.insert(0, j - 1)
            i -= 1
            j -= 1
        elif lcs_table[j][i - 1] > lcs_table[j - 1][i]:
            i -= 1
        else:
            j -= 1
    return backtracked_lcs