import re
import string
from collections import Counter
from typing import Any, Callable, Dict, List, Tuple, Union
from torch import Tensor, tensor
from torchmetrics.utilities import rank_zero_warn
def _metric_max_over_ground_truths(metric_fn: Callable[[str, str], Tensor], prediction: str, ground_truths: List[str]) -> Tensor:
    """Calculate maximum score for a predicted answer with all reference answers."""
    return max((metric_fn(prediction, truth) for truth in ground_truths))