from collections import defaultdict
from itertools import chain
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor, tensor
from torchmetrics.functional.text.helper import _validate_inputs
def _sum_over_dicts(total_n_grams: Dict[int, Tensor], n_grams: Dict[int, Tensor]) -> Dict[int, Tensor]:
    """Aggregate total n-grams to keep corpus-level statistics.

    Args:
        total_n_grams: A dictionary containing a total corpus-level number of n-grams.
        n_grams: A dictionary containing a sentence-level number of n-grams.

    Return:
        A dictionary containing a total corpus-level number of n-grams.

    """
    for n in n_grams:
        total_n_grams[n] += n_grams[n]
    return total_n_grams