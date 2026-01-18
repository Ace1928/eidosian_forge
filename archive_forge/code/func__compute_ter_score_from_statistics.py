import re
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Type, Union
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import (
def _compute_ter_score_from_statistics(num_edits: Tensor, tgt_length: Tensor) -> Tensor:
    """Compute TER score based on pre-computed a number of edits and an average reference length.

    Args:
        num_edits: A number of required edits to match hypothesis and reference sentences.
        tgt_length: An average length of reference sentences.

    Return:
        A corpus-level TER score or 1 if reference_length == 0.

    """
    if tgt_length > 0 and num_edits > 0:
        return num_edits / tgt_length
    if tgt_length == 0 and num_edits > 0:
        return tensor(1.0)
    return tensor(0.0)