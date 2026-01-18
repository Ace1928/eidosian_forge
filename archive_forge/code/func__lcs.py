import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.utilities.imports import _NLTK_AVAILABLE
def _lcs(pred_tokens: Sequence[str], target_tokens: Sequence[str], return_full_table: bool=False) -> Union[int, Sequence[Sequence[int]]]:
    """DP algorithm to compute the length of the longest common subsequence.

    Args:
        pred_tokens: A tokenized predicted sentence.
        target_tokens: A tokenized target sentence.
        return_full_table: If the full table of logest common subsequence should be returned or just the largest

    """
    lcs = [[0] * (len(pred_tokens) + 1) for _ in range(len(target_tokens) + 1)]
    for i in range(1, len(target_tokens) + 1):
        for j in range(1, len(pred_tokens) + 1):
            if target_tokens[i - 1] == pred_tokens[j - 1]:
                lcs[i][j] = lcs[i - 1][j - 1] + 1
            else:
                lcs[i][j] = max(lcs[i - 1][j], lcs[i][j - 1])
    if return_full_table:
        return lcs
    return lcs[-1][-1]